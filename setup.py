# build: python setup.py build_ext --inplace

from setuptools import setup, find_packages# , Extension
# from Cython.Build import cythonize

# relative links to absolute
with open("./README.md", "r") as f:
    readme = f.read()
readme = readme.replace('src="./img_phy_sim/raytracing_example.png"', 'src="https://github.com/M-106/Image-Physics-Simulation/raw/main/img_phy_sim/raytracing_example.png"')
readme = readme.replace('src="./img_phy_sim/ism_example.png"', 'src="https://github.com/M-106/Image-Physics-Simulation/raw/main/img_phy_sim/ism_example.png"')


# ext_1 = Extension(
#     name="img_phy_sim.ray_tracing",
#     sources=["img_phy_sim/ray_tracing.pyx"],
#     include_dirs=[],
#     extra_compile_args=["-O3"],
# )

# ext_2 = Extension(
#     name="img_phy_sim.math",
#     sources=["img_phy_sim/math.pyx"],
#     include_dirs=[],
#     extra_compile_args=["-O3"],
# )

setup(
    # ext_modules=cythonize(
    #     [ext_1, ext_2],
    #     compiler_directives={
    #         "language_level": "3",
    #         "boundscheck": False,
    #         "wraparound": False,
    #         "initializedcheck": False,
    #         "nonecheck": False,
    #         "cdivision": True,
    #     },
    #     annotate=True,
    # ),
    name='img-phy-sim',
    version='1.4',
    packages=find_packages(),  # ['img_phy_sim'],
    install_requires=[
        # List any dependencies here, e.g. 'numpy', 'requests'
        "numpy",
        "opencv-python",
        "matplotlib",
        "scikit-image",
        "joblib",
        "shapely",
        "numba"
    ],
    extras_require={
        # full version (including PhysGen/data module)
        "full": [
            "torch",
            "torchvision",
            "datasets==3.6.0",
            # "prime_printer"
        ]
    },
    author="Tobia Ippolito",
    description = 'Physical Simulations on Images.',
    long_description = readme,
    long_description_content_type="text/markdown",
    include_package_data=True,  # Ensures files from MANIFEST.in are included
    download_url = 'https://github.com/M-106/Image-Physics-Simulation/archive/v_08.tar.gz',
    url="https://github.com/M-106/Image-Physics-Simulation",
    project_urls={
        "Documentation": "https://M-106.github.io/Image-Physics-Simulation/img_phy_sim",
        "Source": "https://github.com/M-106/Image-Physics-Simulation"
    },
    keywords = ['Simulation', 'Computer-Vision', 'Physgen'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',      # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13'
    ],
)