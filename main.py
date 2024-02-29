import ctypes
from dataclasses import dataclass
from enum import Enum
import multiprocessing
from queue import Queue
import subprocess
import sys
import threading
import time
from typing import Union, List

import numpy as np
from numpy.ctypeslib import ndpointer


@dataclass
class Portfolio:
    """
    Container class to keep track of market values weights and sectors for a portfolio.
    """
    market_value_pcts: "np.ndarray"
    sectors: "np.ndarray"


class Sector(Enum):
    """
    Enumeration to store sector information for bonds in a portfolio.
    """
    BANKING = 0
    BASIC_INDUSTRY = 1
    BROKERAGE_ASSET_MANAGERS = 2
    CAPITAL_GOODS = 3
    COMMUNICATIONS = 4
    CONSUMER_CYCLICAL = 5
    CONSUMER_NON_CYCLICAL = 6
    ELECTRIC = 7
    ENERGY = 8
    FINANCE_COMPANIES = 9
    FOREIGN_AGENCIES = 10
    INSURANCE = 11
    NATURAL_GAS = 12
    OTHER_INDUSTRY = 13
    OTHER_UTILITY = 14
    REIT = 15
    TECHNOLOGY = 16
    TRANSPORTATION = 17


def prepare_mock_portfolio(num_bonds: int) -> Portfolio:
    """
    Helper function to prepare some fake portfolio data
    @param num_bonds: how many bonds to put in the fake portfolio
    @return:
    """
    market_value_pcts = np.random.uniform(size=(num_bonds,)).astype(np.float64)
    market_value_pcts /= np.sum(market_value_pcts)
    sectors = np.random.randint(low=0, high=len(
        Sector), size=(num_bonds,), dtype=np.int32)
    portfolio = Portfolio(market_value_pcts, sectors)
    return portfolio


def log_tolerance(worker_num: int, sector: Sector, market_value_pct: float, target_weight: float):
    print(
        f"worker: {worker_num} {sector.name}: {market_value_pct:.3f}, target_weight: {target_weight:.3f}, "
        f"diff_from_tol: {(market_value_pct - target_weight):.3f}\n")


def calculate_sector_weights(worker_num: int, portfolio: Portfolio, target_weight: float):
    """
    Function to calculate how much market value is in each sector and log this for a given portfolio.
    @param worker_num: which worker is responsible for the calculation
    @param portfolio: portfolio container
    @param target_weight: a target weight to compare against market value pct
    @return:
    """
    sector_weights = [0.0] * len(Sector)

    for i in range(portfolio.sectors.shape[0]):
        bond_sector = portfolio.sectors[i]
        sector_weights[bond_sector] += portfolio.market_value_pcts[i]

    for i in range(len(Sector)):
        sector = Sector(i)
        mv_pct = sector_weights[i]
        log_tolerance(worker_num, sector, mv_pct, target_weight)

    return


def iterative_calcs(portfolios: List[Portfolio], target_weight: float):
    """
    Iteratively calculate and log the market value weights for each sector in each portfolio.
    @param portfolios: list of Portfolio instances
    @param target_weight: target sector weight to compare actual weight to
    @return:
    """
    for pf in portfolios:
        calculate_sector_weights(0, pf, target_weight)


def threading_calcs(num_workers: int, portfolios: List[Portfolio], target_weight: float):
    """
    Calculate and log the market value weights for each sector in each portfolio using one thread per portfolio.
    @param num_workers: how many threads to use at a given time. for example, 3.
    @param portfolios: list of Portfolio instances
    @param target_weight: target sector weight to compare actual weight to
    @return:
    """
    for pf_index in range(0, len(portfolios), num_workers):
        worker_list = []
        for thread_index in range(num_workers):
            pf_index = pf_index + thread_index
            if pf_index >= len(portfolios):
                continue
            args = (thread_index, portfolios[pf_index], target_weight)
            worker = threading.Thread(
                target=calculate_sector_weights, args=args)
            worker_list.append(worker)
            worker.start()
        [worker.join() for worker in worker_list]


def multiprocessing_calcs_bad(num_workers: int, portfolios: List[Portfolio], target_weight: float):
    """
    @param num_workers: how many processes to use at a given time. for example, 3.
    @param portfolios: list of Portfolio instances
    @param target_weight: target sector weight to compare actual weight to
    @return:
    """
    for pf_index in range(0, len(portfolios), num_workers):
        worker_list = []
        for thread_index in range(num_workers):
            pf_index = pf_index + thread_index
            if pf_index >= len(portfolios):
                continue
            args = (thread_index, portfolios[pf_index], target_weight)
            worker = multiprocessing.Process(
                target=calculate_sector_weights, args=args)
            worker_list.append(worker)
            worker.start()
        [worker.join() for worker in worker_list]


@dataclass
class MPCalcSectorWeightArgs:
    """
    Container class for arguments to a multiprocessing worker to calculate portfolio weights.
    """
    exit_flag: bool
    pf_index: int


def calc_sector_weights_wrapper(args_queue: Queue, worker_num: int, portfolios: List[Portfolio], target_weight: float):
    """
    Helper function to bring up a worker and keep it up looking for new jobs to run in the queue. Will exit when it
    receives a job with an exit flag set in the queue.
    @param args_queue: a queue to poll for jobs or to exit
    @param worker_num: which worker this is
    @param portfolios: list of Portfolio instances
    @param target_weight: target sector weight to compare actual weight to
    @return:
    """
    exit_flag = False
    while not exit_flag:
        if args_queue.qsize() > 0:
            args = args_queue.get()
            pf_index = args.pf_index
            exit_flag = args.exit_flag
            if exit_flag:
                continue
            calculate_sector_weights(
                worker_num, portfolios[pf_index], target_weight)


def multiprocessing_calcs(num_workers: int, portfolios: List[Portfolio], target_weight: float):
    """
    Calculate and log the market value weights for each sector in each portfolio using the given number of workers and
    keeping these processes up throughout the run to save on process start up time.
    @param num_workers: how many threads to use at a given time. for example, 3.
    @param portfolios: list of Portfolio instances
    @param target_weight: target sector weight to compare actual weight to
    """
    worker_list = []
    q_list = []
    for i in range(num_workers):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=calc_sector_weights_wrapper, args=(q, i, portfolios, target_weight))
        worker_list.append(p)
        q_list.append(q)
    [p.start() for p in worker_list]

    for pf_index in range(0, len(portfolios)):
        args = MPCalcSectorWeightArgs(False, pf_index)
        worker_num = pf_index % num_workers
        q_list[worker_num].put(args)

    for worker_num in range(num_workers):
        args = MPCalcSectorWeightArgs(True, 0)
        q_list[worker_num].put(args)
    [worker.join() for worker in worker_list]


def setup_cpp_lib():
    """
    Helper function to load in the shared library object for the C++ functions and set some basic argument validation
     on the functions in the library. Relies on the .so file being generated and in the working directory
     of the python file.
    @return:
    """
    cpp_lib = ctypes.cdll.LoadLibrary("./pf_sector_helpers.so")
    calculate_sector_weights = cpp_lib.calculate_sector_weights
    calculate_sector_weights.restype = None
    calculate_sector_weights.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_size_t,
                                         ndpointer(
                                             ctypes.c_int32, flags="C_CONTIGUOUS"), ctypes.c_size_t,
                                         ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_size_t]
    return cpp_lib


def cpp_calc_sector_weights(portfolios: List[Portfolio], target_weight: float, cpp_lib):
    """
    Python wrapper function for the C++ function to calculate sector weights. Calculates and logs the market value
     weights for each sector in each portfolio using one thread per portfolio.
    @param portfolios: list of Portfolio instances
    @param target_weight: target sector weight to compare actual weight to
    @param cpp_lib: shared library loaded using setup_cpp_lib
    @return:
    """
    sector_array = np.concatenate(
        [pf.sectors for pf in portfolios], axis=None, dtype=np.int32)
    mv_arrays = np.concatenate(
        [pf.market_value_pcts for pf in portfolios], axis=None, dtype=np.float64)
    if sector_array.shape != mv_arrays.shape:
        raise ValueError("sectors must have same number of entries as bonds")

    num_portfolios = len(portfolios)
    num_sectors = len(Sector)
    num_bonds = portfolios[0].market_value_pcts.shape[0]
    out_arr = np.zeros((num_portfolios * num_sectors,), dtype=np.float64)
    cpp_lib.calculate_sector_weights(
        mv_arrays, num_portfolios, sector_array, num_bonds, out_arr, num_sectors)
    for pf_index in range(num_portfolios):
        row_index = pf_index * num_sectors
        for i in range(num_sectors):
            sector = Sector(i)
            log_tolerance(0, sector, out_arr[row_index + i], target_weight)


def numpy_calc_sector_weights(portfolios: List[Portfolio], target_weight: float):
    """
    Calculates and logs the market value weights for each sector in each portfolio using numpy to achieve parallel
     processing.
    @param portfolios: list of Portfolio instances
    @param target_weight: target sector weight to compare actual weight to
    @return:
    """
    sector_array = np.array([pf.sectors for pf in portfolios], dtype=np.int32)
    mv_arrays = np.array(
        [pf.market_value_pcts for pf in portfolios], dtype=np.float64)
    num_portfolios = len(portfolios)
    num_sectors = len(Sector)
    out_arr = np.zeros((num_portfolios, num_sectors,), dtype=np.float64)

    for sector_index in range(num_sectors):
        sector_mask = np.equal(sector_array, sector_index)
        sector_sums = np.sum(mv_arrays, axis=1, where=sector_mask)
        out_arr[:, sector_index] = sector_sums

    for pf_index in range(num_portfolios):
        for i in range(num_sectors):
            sector = Sector(i)
            log_tolerance(0, sector, out_arr[pf_index, i], target_weight)


def main(method: Union[str, None] = None):
    """
    Function to generate mock data and run it through one of our processing functions.
    @param method: str to specify which method, see strings supported below. Or None to parse the first argument
        given from sys.argv[1]. Example: python3 main.py cpp
    @return:
    """
    start_time = time.time()
    num_workers = 3
    num_bonds = 50000
    num_portfolios = 20
    portfolios = []
    for i in range(num_portfolios):
        portfolios.append(prepare_mock_portfolio(num_bonds))
    target_weight = 1 / len(Sector)
    if method is None:
        method = sys.argv[1]
    print(method)

    if method == "iterative":
        iterative_calcs(portfolios, target_weight)
    elif method == "threading":
        threading_calcs(num_workers, portfolios, target_weight)
    elif method == "multiprocessing-bad":
        multiprocessing_calcs_bad(num_workers, portfolios, target_weight)
    elif method == "multiprocessing":
        multiprocessing_calcs(num_workers, portfolios, target_weight)
    elif method == "cpp":
        cpp_lib = setup_cpp_lib()
        cpp_calc_sector_weights(portfolios, target_weight, cpp_lib)
    elif method == "numpy":
        numpy_calc_sector_weights(portfolios, target_weight)
    else:
        raise ValueError(f"usage: python {__file__} iterative")
    print(f"run_time: {(time.time() - start_time):.5f}")


def monitor_runs():
    """
    Helper to run all the available methods and do basic profiling. Note for the cpp and numpy methods
        the top call isn't fast enough, these need to be profiled in a seperate bash tab.
    @return:
    """
    for run_type in ["iterative", "threading", "multiprocessing-bad", "multiprocessing", "cpp", "numpy"]:
        print()
        print()
        print()
        print()
        print(f"run_type: {run_type}")
        print("-------------------------------------------------")
        tops = []

        top_args = ['top', '-c', '-b', '-n50',
                    '-E', 'm', '-e', 'm', '-d' '0.01']
        p = multiprocessing.Process(target=main, args=(run_type,))
        started = False
        while not started or p.is_alive():
            top_p = subprocess.Popen(top_args, shell=False,
                                     stdout=subprocess.PIPE)
            if not started:
                started = True
                p.start()
            outs, errs = top_p.communicate()
            tops.append(outs.decode())
        [print(top) for top in tops]
        p.join()


if __name__ == "__main__":
    monitor_runs()
