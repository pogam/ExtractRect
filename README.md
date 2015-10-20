# ExtractRect
find the largest rectangle inscribed in a non-convex polygon

the code is inspired from the stackoverflow discussion that can be found here
[here](http://stackoverflow.com/questions/2478447/find-largest-rectangle-containing-only-zeros-in-an-n%C3%97n-binary-matrix/30418912#30418912)

the function findRotMaxRect can be run in different mode using 
* optimization algo
* brute multiprocessor
* brute serial

To speed-up the algo, the initial image was reduce by a factor 3
