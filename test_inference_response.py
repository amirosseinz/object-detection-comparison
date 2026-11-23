import io
import json
import logging
import tempfile
import unittest
from http.cookies import SimpleCookie
from pathlib import Path
from unittest.mock import Mock, patch
from werkzeug.test import Client
from werkzeug.wrappers import Response

from inference_response import parse_detections


def response(boxes=None, scores=None, classes=None, count=None):
    prediction = {
        'detection_boxes': boxes if boxes is not None else [[0.1, 0.2, 0.7, 0.8]],
        'detection_scores': scores if scores is not None else [0.9],
        'detection_classes': classes if classes is not None else [1.0],
    }
    if count is not None:
        prediction['num_detections'] = count
    return {'predictions': [prediction]}


class ResponseValidationTests(unittest.TestCase):
    def test_valid_detections_and_threshold_boundary(self):
        body = response(boxes=[[0, 0, 1, 1]] * 3, scores=[0.9, 0.5, 0.2], classes=[1.0, 2.0, 3.0])
        self.assertEqual(parse_detections(body, 0.5), [([0, 0, 1, 1], 0.9, 1)])

    def test_valid_empty_output(self):
        self.assertEqual(parse_detections(response([], [], []), 0.5), [])

    def test_num_detections_excludes_padding(self):
        body = response(boxes=[[0, 0, 1, 1], []], scores=[0.9, None], classes=[1, None], count=1.0)
        self.assertEqual(len(parse_detections(body, 0.5)), 1)
        self.assertEqual(parse_detections(response(count=0.0), 0.5), [])

    def test_missing_or_wrong_prediction_structure_fails(self):
        for body in (None, [], {}, {'predictions': []}, {'predictions': [None]},
                     {'predictions': [{}]}, {'predictions': [{}, {}]}, {'error': 'Model failed'}):
            with self.subTest(body=body):
                with self.assertRaises(ValueError):
                    parse_detections(body, 0.5)

    def test_inconsistent_lengths_fail_instead_of_truncating(self):
        for field in ('detection_boxes', 'detection_scores', 'detection_classes'):
            body = response()
            body['predictions'][0][field] = []
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    parse_detections(body, 0.5)

    def test_invalid_boxes_scores_classes_and_counts_fail(self):
        for field, values in {
            'detection_boxes': [[[0, 0, 1]], [[0, 0, float('inf'), 1]], [[1, 0, 0, 1]], [[False, 0, 1, 1]]],
            'detection_scores': [[float('nan')], [-0.1], [1.1], [True], ['0.9']],
            'detection_classes': [[1.5], [-1], [True], [float('inf')]],
            'num_detections': [-1, 2, 0.5, True, '1'],
        }.items():
            for value in values:
                body = response()
                body['predictions'][0][field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        parse_detections(body, 0.5)


class InferenceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the real app with only its external model health request mocked.
        logging.disable(logging.CRITICAL)
        with patch('requests.get', return_value=Mock(status_code=200)):
            import app
        cls.module = app

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = patch.dict(self.module.app.config, {
            'RESULTS_DETAILS_FOLDER': str(self.root / 'details'),
            'CLASSIFIED_FOLDER_A': str(self.root / 'model-a'),
            'CLASSIFIED_FOLDER_B': str(self.root / 'model-b'),
            'UPLOAD_FOLDER': str(self.root / 'uploads'),
            'RESULTS_CSV': str(self.root / 'results.csv'),
            'TESTING': True,
            'SECRET_KEY': 'fixture-key',
        })
        self.config.start()
        self.addCleanup(self.config.stop)
        self.image = self.root / 'image.png'
        self.module.Image.new('RGB', (48, 48)).save(self.image)

    def process(self, body):
        upstream = Mock()
        upstream.json.return_value = body
        with patch.object(self.module.requests, 'post', return_value=upstream):
            return self.module.process_image_with_model(
                str(self.image), 'http://model.test/predict', str(self.root / 'output'), 'fixture', 'model_a',
            )

    def test_malformed_response_cannot_create_successful_detection_details(self):
        result = self.process({'predictions': [{}]})
        self.assertFalse(result['success'])
        self.assertFalse(result['details_saved'])
        self.assertIn('requires', result['error'])
        self.assertFalse((self.root / 'details' / 'fixture_model_a.json').exists())

    def test_empty_detection_is_a_successful_inference(self):
        result = self.process(response([], [], []))
        self.assertTrue(result['success'])
        self.assertEqual(result['detection_count'], 0)
        self.assertEqual(json.loads((self.root / 'details' / 'fixture_model_a.json').read_text()), [])

    def test_valid_response_preserves_saved_detection_schema(self):
        result = self.process(response())
        self.assertTrue(result['success'])
        self.assertEqual(result['detections'], [{
            'class_id': 1, 'class_name': self.module.LABEL_MAP[1], 'confidence': 0.9,
            'bbox_normalized': [0.1, 0.2, 0.7, 0.8],
        }])
        self.assertTrue(Path(result['output_filepath']).is_file())

    def test_upload_route_reports_failed_model_to_user(self):
        bad, good = Mock(), Mock()
        bad.json.return_value = {'predictions': [{}]}
        good.json.return_value = response([], [], [])
        with patch.object(self.module.requests, 'post', side_effect=[bad, good]):
            client = Client(self.module.app, Response)
            result = client.post('/inference', data={
                'file': (io.BytesIO(self.image.read_bytes()), 'fixture.png'),
            })
            self.assertEqual(result.status_code, 302)
            cookies = SimpleCookie()
            cookies.load(result.headers['Set-Cookie'])
            serializer = self.module.app.session_interface.get_signing_serializer(self.module.app)
            messages = serializer.loads(cookies['session'].value)['_flashes']
            self.assertTrue(any(kind == 'warning' and 'Model A failed' in message for kind, message in messages))


if __name__ == '__main__':
    unittest.main()
