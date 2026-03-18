"""
This script automates the workflow for modifying the geometry of a plate model in Abaqus, 
running a finite element analysis (FEA) job, and extracting the maximum Von Mises stress 
from the output database (ODB). It also measures and logs the total runtime of the script 
and outputs the results to a specified file.
"""

from abaqus import *
from abaqusConstants import *
from odbAccess import *
from mesh import ElemType
import logging
import sys
import time  # For measuring execution time

print("\n" * 5)  # Print 5 empty lines
print("=" * 160)  # Print a line of 160 asterisks
print("Starting Abaqus FEA Workflow script: MaintenancePlate.py ".center(160))
print("=" * 160)

# Configurable parameters
CAE_NAME = 'MaintenancePlate.cae'
MODEL_NAME = 'Model-1'           # Name of the Abaqus model to modify
PART_NAME = 'Part-1'             # Name of the part in the model
JOB_NAME = 'MaintenancePlate_Job'  # Name of the analysis job
INPUT_FILE = 'MaintenancePlate_StressExtract_Input.txt'  # Input file containing geometry parameters
OUTPUT_FILE = 'MaintenancePlate_StressExtract_Output.txt'  # Output file for results and runtime

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(message)s"  # Only include the log message, no timestamp or log level
)

logging.info("Script started.")

# Function to read input parameters from a file
def read_input(file_path):
    """
    Reads geometry parameters (W1, W2, R, t) from the input file.
    :param file_path: Path to the input file
    :return: List of float values [W1, W2, R, t]
    """
    with open(file_path) as f:
        content = f.readlines()
    return [float(value.strip()) for value in content]
    
    
    #try:
    #    
    #except Exception as e:
    #    print("Error reading input file: {}".format(e))
    #    sys.exit(1)

# Function to modify the geometry of the part and assign section properties
def modify_geometry(model, part, W1, W2, R, t):
    """
    Modifies the geometry of the part based on the input parameters and updates the thickness.
    :param model: Abaqus model object
    :param part: Abaqus part object
    :param W1: Width parameter 1
    :param W2: Width parameter 2
    :param R: Radius parameter
    :param t: Thickness parameter
    """
    print("Modifying geometry for model '{}', part '{}', with W1={}, W2={}, R={}, t={}...".format(
        MODEL_NAME, PART_NAME, W1, W2, R, t))

    # Modify the sketch
    model.ConstrainedSketch(name='__edit__', objectToCopy=part.features['Shell planar-1'].sketch)
    part.projectReferencesOntoSketch(filter=COPLANAR_EDGES, sketch=model.sketches['__edit__'], upToFeature=part.features['Shell planar-1'])
    model.sketches['__edit__'].parameters['W1'].setValues(expression=str(W1))
    model.sketches['__edit__'].parameters['W2'].setValues(expression=str(W2))
    model.sketches['__edit__'].parameters['R'].setValues(expression=str(R))
    part.features['Shell planar-1'].setValues(sketch=model.sketches['__edit__'])
    del model.sketches['__edit__']
    part.regenerate()

    # Check geometry validity
    part.checkGeometry()
    print("Geometry checked and is valid.")

    # Update thickness and reassign section
    model.sections['Section-1'].setValues(thickness=t)
    if not any(assignment.sectionName == 'Section-1' for assignment in part.sectionAssignments):
        all_cells = part.cells
        region = part.Set(cells=all_cells, name='AllCells')
        part.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, offsetType=MIDDLE_SURFACE)
        print("Section 'Section-1' assigned to all cells.")
    else:
        print("Section 'Section-1' is already assigned. Skipping reassignment.")

    # Clear existing mesh and remesh
    part.deleteMesh()
    print("Previous mesh deleted.")
    part.seedPart(size=0.005, deviationFactor=0.1, minSizeFactor=0.1)
    part.setElementType(elemTypes=(ElemType(
        elemCode=S8R, elemLibrary=STANDARD), ElemType(elemCode=STRI65, elemLibrary=STANDARD)), regions=(
        part.faces.getSequenceFromMask(('[#1 ]',), ), ))
    print("Mesh seeding applied.")
    part.generateMesh()
    print("New mesh generated.")

    # Regenerate the part and assembly
    part.regenerate()
    model.rootAssembly.regenerate()
    print("Part and assembly regenerated.")

# Function to submit an Abaqus job
def submit_job(mdb,job_name):
    """
    Submits the Abaqus job and waits for it to complete.
    :param job_name: Name of the job to submit
    """
    print("Submitting job '{}'...".format(job_name))
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    mdb.jobs[job_name].waitForCompletion()
    print("Job '{}' completed successfully.".format(job_name))

# Function to extract results from the ODB file and write to output
def extract_results(job_name, output_file, start_time):
    """
    Extracts the maximum Von Mises stress from the ODB file and writes the results and runtime to a file.
    :param job_name: Name of the job to retrieve results from
    :param output_file: Path to the output file
    :param start_time: Start time of the script (to calculate total runtime)
    """
    try:
        odb = openOdb(path="{}.odb".format(job_name))
        step = odb.steps.values()[0]
        step_values = step.frames[1].fieldOutputs['S']
        von_mises = [value.mises for value in step_values.values]
        von_mises_max = max(von_mises)/1e6
        print("Max Von Mises Stress (MPa): {:.8f}".format(von_mises_max))

        # Calculate total runtime
        total_time = time.time() - start_time
        print("Script runtime: {:.8f} seconds.".format(total_time))

        # Write results to the output file
        with open(output_file, 'w') as file_output:
            file_output.write("Max Von Mises Stress: {:.8f} MPa\n".format(von_mises_max))
            file_output.write("Script Runtime: {:.8f} seconds\n".format(total_time))
        print("Results written to '{}'.".format(output_file))
    finally:
        odb.close()

# Main workflow
start_time = time.time()  # Start the timer
mdb = openMdb(CAE_NAME)

# Step 1: Read input parameters
input_values = read_input(INPUT_FILE)

# Step 2: Modify geometry
modify_geometry(mdb.models[MODEL_NAME], mdb.models[MODEL_NAME].parts[PART_NAME], *input_values)

# Step 3: Submit the job
submit_job(mdb,JOB_NAME)

# Step 4: Extract results and write to the output file
extract_results(JOB_NAME, OUTPUT_FILE, start_time)

logging.info("Script finished successfully.")
