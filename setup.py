import setuptools

with open("README.md", "r", encoding='utf-8') as readme_file:
    long_description = readme_file.read()

with open('requirements.txt', 'r', encoding='utf-8') as req_file:
    requirements = [r.strip() for r in req_file.readlines()]

setuptools.setup(
    name="quantum_decomp",
    version="1.1.1",
    author="Dmytro Fedoriaka",
    description="Tool for decomposing unitary matrix into quantum gates",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    url="https://github.com/fedimser/quantum_decomp",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=requirements,
)
