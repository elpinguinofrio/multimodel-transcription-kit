"""Launch the packaged core with paths resolved on the consuming machine."""
import configparser
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import uuid


def prepare_config(root, bundle, machine=None):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    relative = Path(manifest['env_relative'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Portable env file must be relative to the shared root')
    machine = re.sub(r'[^A-Za-z0-9_.-]', '_', machine or platform.node())
    state = root / '.private' / 'portable-state' / machine
    state.mkdir(parents=True, exist_ok=True)
    cfg = configparser.ConfigParser()
    cfg.read(bundle / 'config.ini')
    for section in ('secrets', 'limits', 'usage', 'defaults'):
        if not cfg.has_section(section):
            cfg.add_section(section)
    cfg['secrets']['env_file'] = str(root.parents[1] / relative)
    cfg['limits']['state_file'] = str(state / 'ratelimit.sqlite')
    cfg['usage']['file'] = str(state / 'usage.sqlite')
    cfg['defaults']['enhanced_dir'] = str(state / 'enhanced')
    target = state / f'config-{uuid.uuid4().hex}.ini'
    with target.open('x', encoding='utf-8') as output:
        cfg.write(output)
    return target


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(__file__).resolve().parents[1]
    bundle = root / '.private' / 'portable-macos-arm64-kit'
    sys.path.insert(0, str(bundle / 'site-packages'))
    if argv == ['--portable-info']:
        import multimodel_transcription_kit as core
        manifest = json.loads((bundle / 'manifest.json').read_text())
        print(json.dumps({
            'machine': platform.node(), 'python': sys.executable,
            'python_prefix': sys.prefix, 'package': core.__file__,
            'shared_root': str(root.parents[1]),
            'env_file_exists': (root.parents[1] / manifest['env_relative']).is_file(),
            'ffmpeg': shutil.which('ffmpeg'), 'ffprobe': shutil.which('ffprobe'),
        }, indent=2))
        return 0
    # The packaged CLI is the only workflow. Explicit caller configuration wins.
    help_only = any(arg in ('-h', '--help') for arg in argv)
    explicit = any(arg == '--config' or arg.startswith('--config=') for arg in argv)
    if not help_only:
        import tempfile
        work = root / '.private' / 'portable-tmp'
        work.mkdir(parents=True, exist_ok=True)
        os.environ['TMPDIR'] = str(work)
        tempfile.tempdir = str(work)
    if not help_only and not explicit and (not argv or argv[0] != 'http'):
        argv.extend(['--config', str(prepare_config(root, bundle))])
    from multimodel_transcription_kit.cli import main as cli_main
    return cli_main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
