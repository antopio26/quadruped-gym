class BaseControls:
    def __init__(self):
        pass
        
    def get_obs(self):
        """
        Returns the observation from the environment.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def sample():
        """
        Returns a sample from the environment.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")