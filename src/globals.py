"""Global variables for the trading platform"""

# Global collector instance
collector = None

def set_collector(collector_instance):
    """Set the global collector instance"""
    global collector
    collector = collector_instance

def get_collector():
    """Get the global collector instance"""
    global collector
    return collector
