
By Warren Woodrich Pettine, M.D.
Last updated 2023-11-18

For questions, contact warren.pettine@hsc.utah.edu

## Installation
The code was written in python 3.9. Packages are listed in requirements.txt. The following lines will create a local virtual environment.

```
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## File descriptions

# Notebooks
- make_drone_stims.ipynb organizes functions to programatically generate drone stimuli. These stimuli are used in the online game.
- prolific_study_management.ipynb provides an interface for launching studies on Prolific. It is non-specific to the drone_recon game, and can be independently adapted for any webapp. 
- task_analysis.ipynb steps through analyses of subject responses to questionnaires, along with task behavior. 

# Function packages
- create_drone_stims.py contains functions for the programatic creation of drone stimuli. These could easily be integrated into the webapp, if one wants to generate stimuli on the fly for subjects.
- dbinterfacefunctions.py contains functions for interacting with the database, specifically in the case of the Django webapp. With minor reworking, these are generalizable to other webapps.
- prolificinterfacefunctions.py contains functions for creating and accessing Prolific studies. 
- analysisfunctions.py contains general functions used in analysis. 