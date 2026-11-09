"""ffmpeg/ffprobe print file metadata verbatim; old recordings carry non-UTF-8 tag bytes.
Reading that output must never crash the kit (seen on 2016 Telegram voice messages)."""
import os
import stat
import tempfile
import unittest
from pathlib import Path

from multimodel_transcription_kit import audio_quality
from multimodel_transcription_kit.audio_converter import AudioConverter

BAD = b"\x90\xc0"   # not valid UTF-8

FAKE = """#!/bin/sh
printf 'RMS_level=-20.0\\nRMS_level=-60.0\\n'
printf 'title: \\220\\300 junk\\n{"input_i": "-20.0", "input_tp": "-3.0"}\\n' >&2
"""
FAKE_PROBE = """#!/bin/sh
printf '12.5\\n'
printf 'tag \\220\\300\\n' >&2
"""


class NonUtf8FfmpegOutput(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        for name, body in (("ffmpeg", FAKE), ("ffprobe", FAKE_PROBE)):
            p = d / name
            p.write_text(body)
            p.chmod(p.stat().st_mode | stat.S_IEXEC)
        self.audio = d / "a.ogg"
        self.audio.write_bytes(b"x")
        self.old_path = os.environ["PATH"]
        os.environ["PATH"] = "%s%s%s" % (d, os.pathsep, self.old_path)

    def tearDown(self):
        os.environ["PATH"] = self.old_path
        self.tmp.cleanup()

    def test_assess_survives_non_utf8_stderr(self):
        result = audio_quality.assess(self.audio)
        self.assertEqual(result["integrated_lufs"], -20.0)

    def test_duration_survives_non_utf8_stderr(self):
        self.assertEqual(AudioConverter.get_audio_duration(str(self.audio)), 12.5)


if __name__ == "__main__":
    unittest.main()
