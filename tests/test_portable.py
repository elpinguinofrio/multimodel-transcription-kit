"""Host launch must not depend on the VM venv, PATH, or VM home config."""
import configparser
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / 'scripts' / f'{name}.py'
    assert path.is_file(), f'{name} must exist for host/VM execution'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_portable_config_uses_current_shared_mount_and_keeps_source(tmp_path):
    entry = load_script('portable_entry')
    root = tmp_path / 'host shared path' / 'projects' / 'core'
    bundle = root / '.private' / 'portable-macos-arm64-kit'
    bundle.mkdir(parents=True)
    original = '[secrets]\nenv_file = ~/.config/transcription/.env\n[models]\nranking = a\n[model.a]\nkind = gemini\n'
    (bundle / 'config.ini').write_text(original)
    (bundle / 'manifest.json').write_text(json.dumps({'env_relative': '.env/.env'}))
    cfg_path = entry.prepare_config(root, bundle, machine='host-one')
    cfg = configparser.ConfigParser(); cfg.read(cfg_path)
    assert Path(cfg['secrets']['env_file']) == root.parents[1] / '.env' / '.env'
    assert Path(cfg['limits']['state_file']).is_relative_to(root / '.private' / 'portable-state' / 'host-one')
    assert Path(cfg['defaults']['enhanced_dir']).is_relative_to(root / '.private')
    assert cfg['models']['ranking'] == 'a'
    assert (bundle / 'config.ini').read_text() == original


def test_missing_bundle_does_not_execute_broken_vm_venv(tmp_path):
    launcher = ROOT / 'multimodel-transcription-kit'
    assert launcher.is_file(), 'A repository-relative launcher is required'
    project = tmp_path / 'host mount' / 'core'; project.mkdir(parents=True)
    shutil.copy2(launcher, project / 'multimodel-transcription-kit')
    old = project / '.venv' / 'bin'; old.mkdir(parents=True)
    marker = project / 'wrong-runtime-used'
    (old / 'python').write_text(f'#!/bin/sh\ntouch "{marker}"\n')
    (old / 'python').chmod(0o755)
    result = subprocess.run(['/bin/sh', str(project / 'multimodel-transcription-kit'), '--help'], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'portable' in result.stderr.lower()
    assert not marker.exists()


def test_portable_cli_keeps_temporary_audio_in_the_shared_checkout(tmp_path, monkeypatch):
    entry = load_script('portable_entry')
    project = tmp_path / 'host mount' / 'projects' / 'core'
    monkeypatch.setattr(entry, '__file__', str(project / 'scripts' / 'portable_entry.py'))
    monkeypatch.setattr(tempfile, 'tempdir', None)
    monkeypatch.setenv('TMPDIR', str(tmp_path / 'outside-share'))
    monkeypatch.syspath_prepend(str(tmp_path))
    from multimodel_transcription_kit import cli

    def run(argv):
        assert argv == ['recording.wav', '--config', 'explicit.ini']
        assert Path(tempfile.gettempdir()).is_relative_to(project / '.private')
        assert os.environ['TMPDIR'] == tempfile.gettempdir()
        return 0

    monkeypatch.setattr(cli, 'main', run)
    assert entry.main(['recording.wav', '--config', 'explicit.ini']) == 0


def test_builder_refuses_an_existing_destination(tmp_path):
    builder = load_script('build_portable')
    output = tmp_path / 'owned'; output.mkdir()
    sentinel = output / 'keep'; sentinel.write_text('unchanged')
    with pytest.raises(FileExistsError):
        builder.build(wheel=tmp_path / 'unused.whl', output=output,
                      python_runtime=tmp_path / 'unused-python', config=tmp_path / 'unused.ini',
                      shared_root=tmp_path, env_file=tmp_path / '.env')
    assert sentinel.read_text() == 'unchanged'


def test_builder_rejects_wheel_path_traversal(tmp_path):
    builder = load_script('build_portable')
    wheel = tmp_path / 'bad.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('../escaped', 'unsafe')
    target = tmp_path / 'target'; target.mkdir()
    with pytest.raises(ValueError, match='wheel'):
        builder.unpack_wheel(wheel, target)
    assert not (tmp_path / 'escaped').exists()


def test_relocated_launcher_runs_cli_with_empty_home_and_minimal_path(tmp_path):
    builder = load_script('build_portable')
    project = tmp_path / 'different host mount' / 'projects' / 'core'
    scripts = project / 'scripts'; scripts.mkdir(parents=True)
    shutil.copy2(ROOT / 'multimodel-transcription-kit', project / 'multimodel-transcription-kit')
    shutil.copy2(ROOT / 'scripts' / 'portable_entry.py', scripts / 'portable_entry.py')
    bundle = project / '.private' / 'portable-macos-arm64-kit'
    binary = bundle / 'python' / 'bin'; binary.mkdir(parents=True)
    (binary / 'python3').symlink_to(sys.executable)
    site = bundle / 'site-packages'; site.mkdir()
    shutil.copytree(ROOT / 'src' / 'multimodel_transcription_kit', site / 'multimodel_transcription_kit',
                    ignore=shutil.ignore_patterns('__pycache__'))
    for name in builder.DEPENDENCIES:
        builder.copy_distribution(name, site)
    home = tmp_path / 'empty-home'; home.mkdir()
    env = dict(os.environ, HOME=str(home), PATH='/usr/bin:/bin', PYTHONPATH='/nonexistent')
    result = subprocess.run(['/bin/sh', str(project / 'multimodel-transcription-kit'), '--help'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'Transcribe one audio file' in result.stdout
    assert not list(home.iterdir())
