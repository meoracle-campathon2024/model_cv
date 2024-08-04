import flwr
from flwr.server import ServerApp, ServerConfig, ServerAppComponents, start_server
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context
from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping

strategy = FedAvg(
    fraction_fit=1,  
    fraction_evaluate=1, 
    min_fit_clients=1,  
    min_evaluate_clients=1, 
    min_available_clients=1,  
)

config = ServerConfig(num_rounds=5)

start_server(server_address="0.0.0.0:8080", config=config, strategy=strategy)

