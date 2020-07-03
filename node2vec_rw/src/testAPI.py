import configparser
import argparse 
import os
from  src.node2vec import node2vec, main



args=argparse.ArgumentParser()
args.add_argument('--download-data', type=str, dest='download', help='download_data' )
args.add_argument('--check-data', type=str, dest='check', help='checks-data')    
parsed = args.parse_args()
parsed_values = [parsed.check, parsed.download]


"""
For now display just the values
TODO: Make some cool cli interface for the node2vec api

"""
def main(args, names : list()):
    for name in names:
        for vals in args.EXECUTE_ATTR(names):
                pass    

