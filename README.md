# TicTacML

#Dependancy

+ [Theano](http://www.github.com/Theano/Theano)
+ [Blocks](https://github.com/mila-udem/blocks)
+ [Pastalog](https://github.com/rewonc/pastalog)

#Reading Material

+ Sutton's book
+ [Karpathy's blog](http://karpathy.github.io/2016/05/31/rl/)
+ [David Silver's lecture on Policy Gradient](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf)

#Current Version

__Right now it supports training of an Policy Agent with a Random Agent and then beating it__
As long as drawing of the game was not being penalized, the Policy agent didnt learn to beat the random agent, however, I then introduced a term to give a negative penaly if the game was drawed.

![Progress](https://raw.githubusercontent.com/amartya18x/TicTacML/master/progress.png "Progress")

__Added them to play against each other __
On retaining the penalty for drawing the game, the number of draws eventually went to zeros and both the players won equal number of games. So, I decided to see what happens if I remove the penalty from draws and then they decided to draw all the games. Interesting eh ?

![All Draws](https://raw.githubusercontent.com/amartya18x/TicTacML/master/draw.png "Draws")
