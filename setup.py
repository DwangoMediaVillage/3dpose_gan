from setuptools import setup, find_packages

def requirements():
    list_requirements = []
    with open('requirements.txt') as f:
        for line in f:
            list_requirements.append(line.rstrip())
    return list_requirements

setup(
    name='projection_gan',
    version='0.0.1',
    packages=['projection_gan'],
    url='https://github.com/DwangoMediaVillage/3dpose_gan',
    author='Yasunori Kudo, and Keisuke Ogaki',
    author_email='keisuke_ogaki@dwango.co.jp',
    description='Generate 3D-pose from 2D-pose.',
    install_requires=requirements(),
    test_suite='tests',
)
