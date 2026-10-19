"""Assemble existing Python, installed dependencies and a wheel; no downloads.

Build into a new private shared-directory destination. Never reuse a virtualenv
across host/VM mount spellings. Nothing is installed into either machine's home.
"""
import argparse
import configparser
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path
import shutil
import sys
import zipfile

DEPENDENCIES = ('requests', 'python-dotenv', 'websockets', 'certifi',
                'charset-normalizer', 'idna', 'urllib3')


def copy_distribution(name, site):
    dist = distribution(name)
    for relative in dist.files or []:
        relative = Path(str(relative))
        # Wheel scripts such as ../../../bin/dotenv are not part of site-packages.
        if relative.is_absolute() or '..' in relative.parts or relative.suffix == '.pyc':
            continue
        source = Path(dist.locate_file(relative))
        if source.is_file():
            target = site / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return dist.version


def unpack_wheel(wheel, target):
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            path = Path(member.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in member.filename:
                raise ValueError('Unsafe wheel member path')
        archive.extractall(target)


def build(*, wheel, output, python_runtime, config, shared_root, env_file):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to replace an existing bundle: {output}')
    python_runtime = Path(python_runtime).resolve(strict=True)
    if not (python_runtime / 'bin' / 'python3').is_file():
        raise ValueError('Expected an existing standalone Python runtime, not a venv')
    # A copied absolute symlink would preserve the original VM dependency.
    for path in python_runtime.rglob('*'):
        if path.is_symlink() and Path(path.readlink()).is_absolute():
            raise ValueError(f'Runtime has a non-relocatable absolute symlink: {path}')
    shared_root = Path(shared_root).resolve(strict=True)
    env_relative = Path(env_file).resolve(strict=True).relative_to(shared_root)
    cfg = configparser.ConfigParser()
    with Path(config).open(encoding='utf-8') as source:
        cfg.read_file(source)
    # Keys are read at execution from the explicitly selected shared env file.
    # Never embed credentials into the bundle's configuration template.
    for section in cfg.sections():
        for key in cfg[section]:
            if key.lower() in {'api_key', 'token', 'password', 'secret'}:
                raise ValueError('Config contains an inline secret; use the shared env file')
    output.mkdir(parents=True)
    shutil.copytree(python_runtime, output / 'python', symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    site = output / 'site-packages'
    site.mkdir()
    unpack_wheel(wheel, site)
    versions = {name: copy_distribution(name, site) for name in DEPENDENCIES}
    with (output / 'config.ini').open('x', encoding='utf-8') as handle:
        cfg.write(handle)
    manifest = {'schema': 1, 'env_relative': str(env_relative),
                'python_version': sys.version.split()[0], 'dependencies': versions,
                'wheel_sha256': hashlib.sha256(Path(wheel).read_bytes()).hexdigest()}
    with (output / 'manifest.json').open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
    print(f'Portable bundle assembled: {output}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('wheel', 'output', 'python-runtime', 'config', 'shared-root', 'env-file'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    build(**vars(args))


if __name__ == '__main__':
    main()
