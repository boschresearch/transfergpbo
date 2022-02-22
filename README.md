# TransferGPBO

This is the companion code for the benchmarking study reported in the paper
Transfer Learning with Gaussian Processes for Bayesian Optimization 
by Petru Tighineanu et al. The paper can be found at 
https://arxiv.org/abs/2111.11223 (will be replaced with the AISTATS link once available) 
and was accepted for publication at AISTATS 2022. The code allows the users to 
reproduce and extend the results reported in the study. Please cite the above paper 
when reporting, reproducing or extending the results.

## Purpose of the project
This software is a research prototype, solely developed for and published as
part of the publication [cited above](https://arxiv.org/abs/2111.11223). It will 
neither be maintained nor monitored in any way.

## Installation
The code was tested with Python 3.8. Install inside your virtual environment as
```bash
pip install .
```

## Running experiments
A Bayesian optimization experiment with a configuration
specified inside `transfergpbo/parameters.py`
can be run via
```bash
python transfergpbo/experiment.py
```
 
## License
TransferGPBO is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in TransferGPBO, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).

