import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from multimodel_transcription_kit import GroqAPI


class WordTimesTest(unittest.TestCase):
    def test_verbose_transcription_requests_real_word_and_segment_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / 'a.wav'
            audio.write_bytes(b'RIFF' + b'\0' * 64)
            words = [{'word': 'Привет', 'start': 0.1, 'end': 0.7}]
            response=MagicMock(status_code=200)
            response.json.return_value={'text':'Привет','words':words}
            with patch.dict('os.environ', {'GROQ_API_KEY':'unit-test-key'}), \
                 patch('requests.post', return_value=response) as post, \
                 patch('multimodel_transcription_kit.audio_converter.AudioConverter.get_audio_duration', return_value=1.0):
                result = GroqAPI().speech_to_text(str(audio), response_format='verbose_json')
            self.assertEqual(post.call_args.kwargs['data']['timestamp_granularities[]'], ['word', 'segment'])
            self.assertEqual(result['raw_response']['words'],words)
