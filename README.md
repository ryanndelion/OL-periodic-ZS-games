# Online Learning in Periodic Zero-Sum Games
This is a repository for code which was used to produce experiments in the paper Online Learning in Periodic Zero-Sum Games, which is to appear at NeurIPS 2021. 
You can find the arxiv of the paper here: [TO ADD ARXIV LINK]

## Abstract
A seminal result in game theory is von Neumann's minmax theorem, which states that zero-sum games admit an essentially unique equilibrium solution. 
Classical learning results build on this theorem to show that online no-regret dynamics converge to an equilibrium in a time-average sense in zero-sum games. 
In the past several years, a key research direction has focused on characterizing the transient behavior of such dynamics. 
General results in this direction show that broad classes of online learning dynamics are cyclic, and formally Poincaré recurrent, in zero-sum games. 
We analyze the robustness of these online learning behaviors in the case of periodic zero-sum games with a time-invariant equilibrium. 
This model generalizes the usual repeated game formulation while also being a realistic and natural model of a repeated competition between players that depends on exogenous environmental variations such as time-of-day effects, week-to-week trends, and seasonality. 
Interestingly, time-average convergence may fail even in the simplest such settings, in spite of the equilibrium being fixed. 
In contrast, using novel analysis methods, we show that Poincaré recurrence provably generalizes despite the complex, non-autonomous nature of these dynamical systems.

## Codebase
The paper presents several simulations in both two-player and polymatrix (multiplayer) games. All experiments performed for this work were done using Python 3.7 and have been compiled intoa Jupyter notebook for ease 
of viewing. Running the code requires only basic scientific computing packages such as NumPy and SciPy, as well as data visualization packages such as Matplotlib and Plotly. 
Most of the code in our submission has been edited such that it can be easily executed on a standard computer in a matter of minutes.

### Requirements
To install requirements:

```pip install -r requirements.txt```
