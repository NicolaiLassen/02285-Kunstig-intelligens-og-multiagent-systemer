/*******************************************************\
|                AI and MAS: Searchclient               |
|                        README                         |
\*******************************************************/


```
# Run server on level
java -jar server.jar -l levels/G0.lvl -c "python main_server.py" -g -s 150 -t 180

# Run server on competition levels
java -jar server.jar -c "python main_server.py" -l "levels_comp" -t 180 -o "output/46.zip"

```















This readme describes how to use the included Python searchclient with the server that is contained in server.jar. 

The Python search client requires at least Python version 3.7, and has been tested with CPython.
The search client requires the 'psutil' package to monitor its memory usage; the package can be installed with pip:
    $ pip install psutil

All the following commands assume the working directory is the one this readme is located in.

You can read about the server options using the -h argument:
    $ java -jar ../server.jar -h

Starting the server using the searchclient:
    $ java -jar ../server.jar -l ../levels/SAD1.lvl -c "python searchclient/searchclient.py" -g -s 150 -t 180

The searchclient uses the BFS search strategy by default. Use arguments -dfs, -astar, -wastar, or -greedy to set alternative search strategies (after you implement them). For instance, to use DFS search on the same level as above:
    $ java -jar ../server.jar -l ../levels/SAD1.lvl -c "python searchclient/searchclient.py -dfs" -g -s 150 -t 180

Memory settings:
    * Unless your hardware is unable to support this, you should let the searchclient allocate at least 2GB of memory *
    The searchclient monitors its own process' memory usage and terminates the search if it exceeds a given memory threshold.
    To set the max memory usage to 2GB (which is also the default):
        $ java -jar ../server.jar -l ../levels/SAD1.lvl -c "python searchclient/searchclient.py --max-memory 2048" -g -s 150 -t 180
    Avoid setting max memory usage too high, since it will lead to your OS doing memory swapping which is terribly slow.

Rendering on Unix systems:
    You may experience poor performance when rendering on some Unix systems, because hardware rendering is not enabled by default.
    To enable OpenGL hardware acceleration you should use the following JVM option: -Dsun.java2d.opengl=true
        $ java -Dsun.java2d.opengl=true -jar ../server.jar -l ../levels/SAD1.lvl -c "python searchclient/searchclient.py" -g -s 150 -t 180
