from setuptools import setup

package_name = 'stretch_audio_interface'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # install launch + template files
        ('share/' + package_name + '/launch', ['launch/audio_feedback.launch.py']),
        ('share/' + package_name + '/config', ['config/audio_feedback.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your@email.com',
    description='Audio feedback interface using Coqui TTS',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'audio_feedback_node = stretch_audio_interface.audio_feedback_node:main'
            ],
    },
)
