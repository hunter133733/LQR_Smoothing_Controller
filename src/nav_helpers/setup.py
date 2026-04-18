from setuptools import setup

package_name = "nav_helpers"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Mo Chen",
    maintainer_email="mochen@cs.sfu.ca",
    description="Nav helpers with state-action trajectories",
)
