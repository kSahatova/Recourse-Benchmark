from setuptools import setup 
  
setup( 
    name='carla-visual', 
    version='0.1', 
    description='An adapted CARLA package for recourse methods to generate counterfactual explanations for images', 
    author='Kseniya Sahatova', 
    author_email='sahatova.kseniya@gmail.com', 
    packages=["carla_visual", "carla_visual.dataloaders",
              "carla_visual.evaluation", "carla_visual.models",
              "carla_visual.plotting", "carla_visual.recourse_methods"], 
    include_package_data=True 
) 