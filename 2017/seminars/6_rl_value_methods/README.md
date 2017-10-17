#### __Lecture slides__ - [here](https://github.com/neychev/ml-mipt-part2/blob/master/2017/lectures/6_temporal_difference.pdf)
### Materials
* Russian materials:
   - Recap of value iteration & Q_learning - [video](https://yadi.sk/i/cVawsPkK3EtGJj)
   - Q-learning seminar - [video](https://yadi.sk/i/dQmolwOy3EtGNK)
   - More on value-based methods - [video 2](https://yadi.sk/i/I7XcP6vU3ExNrT)
   - Sarsa & stuff - [seminar2](https://yadi.sk/i/XbqNQmjm3ExNsq)
* English materials:
   - Lecture by Pieter Abbeel (english) - [video](https://www.youtube.com/watch?v=ifma8G7LegE)
   - Alternative lectures by David Silver (english) - [video part I](https://www.youtube.com/watch?v=Nd1-UUMVfz4) [video part II](https://www.youtube.com/watch?v=PnHCvfgC_ZA), [video part III](https://www.youtube.com/watch?v=0g4j2k_Ggc4&t=43s)
   - Alternative lecture by John Schulmann (english) - [video](https://www.youtube.com/watch?v=IL3gVyJMmhg)
   - Blog post on q-learning Vs SARSA - [url](https://studywolf.wordpress.com/2013/07/01/reinforcement-learning-sarsa-vs-q-learning/)

### More materials
* N-step temporal difference from Sutton's book - [url](http://incompleteideas.net/sutton/book/ebook/node73.html)
* Eligibility traces from Sutton's book - [url](http://incompleteideas.net/sutton/book/ebook/node72.html)
* Blog post on eligibility traces - [url](http://pierrelucbacon.com/traces/)

### Coding practice

This week's practice will require you to pick __either ./seminar_main (python2,local machine) or ./seminar_alternative(otherwise)__.

### ./seminar_main
_this assignment borrows code from awesome [cs188](http://ai.berkeley.edu/project_overview.html)_
This homework assignment works on __python2 only__. If you stick to py3, consider seminar_alternative. Or just install it for this homework alone and remove afterwards.

This homework also requires some physical display (e.g. laptop monitor). It won't work on binder VM / headless server. Please run it on laptop or consider ./seminar_alternative


* You need to implement **QLearining** algorithm.  If you're running go to ```seminar_main/``` folder and open file ```qlearningAgent.py```.

Once you're done, run use those commands:
```
python crawler.py # Crawler with qlearning
python pacman.py -p <your agent> -x <number of train samples> -n <total number of samples> -l <grid env>
python pacman.py -p PacmanQAgent -x 5000 -n 5010 -l smallGrid # example
```
* Make sure you can tune agent to beat ./run_crawler.sh
 * on windows, just run `python crawler.py` from cmd in the project directory
* other ./run* files are mostly for your amusement. 
  * ./run_pacman.sh will need more epochs to converge, see [comments](https://github.com/yandexdataschool/Practical_RL/blob/fall17/week3/seminar_main/run_pacman.sh)
  * on windows, just copy the type `python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid` in cmd from assignemnt dir
(YSDA/HSE) Please submit only qlearningAgents.py file and include a brief text report as comments in it.

### ./seminar_alternative

You'll have to implement qlearning.py just like in main seminar, but in ./seminar_alternative folder. After you're done with it, open the seminar notebook and follow instructions from there.

