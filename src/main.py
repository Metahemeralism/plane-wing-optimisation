if __name__ == "__main__":
    import sys
    import logging
    from src.MaintenancePlate_StressExtract import read_input, modify_geometry
    
    # Check for input file argument
    if len(sys.argv) != 2:
        print("Usage: python main.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Read input parameters
    try:
        W1, W2, R, t = read_input(input_file)
        logging.info("Input parameters read successfully: W1={}, W2={}, R={}, t={}".format(W1, W2, R, t))
    except Exception as e:
        logging.error("Error reading input file: {}".format(e))
        sys.exit(1)
    
    # Modify geometry and assign section properties
    try:
        # Assuming 'model' and 'part' are defined in the context of the Abaqus environment
        modify_geometry(model, part, W1, W2, R, t)
        logging.info("Geometry modified and section properties assigned successfully.")
    except Exception as e:
        logging.error("Error modifying geometry or assigning section properties: {}".format(e))
        sys.exit(1)