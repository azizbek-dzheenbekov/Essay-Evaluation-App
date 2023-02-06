import sys   
import hydra
from logger.logger import setup_applevel_logger

from src.Orchestrator import Orchestrator
from src.LinearEmbeddingsHandler import LinearEmbeddingsHandler
from src.LinearFeaturesHandler import LinearFeaturesHandler

log = setup_applevel_logger(file_name="logger/app_debug.log")

@hydra.main(config_path='config', config_name='config.yaml', version_base=None)
def main(cnf):
    # Log a message to indicate that the main function has been called
    log.debug("Starting main function")
    
    # Choose the model handler based on the value specified in the config file
    model_handler_name = cnf.main.modelHandler
    
    # Log the main parameters taken from the config
    log.info("modelHandler: %s", model_handler_name)
    log.info("preprocess: %s", cnf.main.preprocess)
    log.info("train: %s", cnf.main.train)
    log.info("load: %s", cnf.main.load)
    log.info("save: %s", cnf.main.save)
    log.info("predict: %s", cnf.main.predict)
    
    # Create an instance of the appropriate model handler class
    try:
        # Log a message to indicate that the model handler is being instantiated
        log.debug("Instantiating model handler")
        
        if model_handler_name == "LinearFeaturesHandler":
            log.debug("Running features model")
            Orchestrator(cnf, LinearFeaturesHandler).orchestrate()
            
            
        elif model_handler_name == "LinearEmbeddingsHandler":
            log.debug("Running embeddings model")
            Orchestrator(cnf, LinearEmbeddingsHandler).orchestrate()
            
    except Exception as e:
        log.error("An error occurred while instantiating the model handler: %s", e)
        sys.exit(1)
        
    # Log a message to indicate that the main function has finished executing
    log.debug("Finished main function")

def print_menu():
    print()
    print('|-----------------------------------------|')
    print('|          Essay Evaluation Tool          |')
    print('|-----------------------------------------|')
    
if __name__ == "__main__":
    print_menu()
    main()
