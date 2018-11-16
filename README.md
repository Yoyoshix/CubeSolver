# CubeSolver
YOU HAVE TETRIS PEIECE AND YOU WANT TO CREATE A CUBE SHAPE WITH THESE PEIECES \

# What needs to be implemented ?
- Upload the minimalist version. A version that works with the minimal amount of line code, like the windows' safe mod. Useful to better understand the algorithm. \
- Fix the forbidden feature. Idea is to update the forbidden pieces matrix once a solution is found. Actual function is wrongly updated and I think it causes the current issue which "skip" solutions. \
- Fix the rotsym feature. Idea is to work with all different interaction between pieces. For a given 3D shape the amount of interactions is : 3\*shape0\*shape1\*shape2 - shape0\*shape1 - shape0\*shape2 - shape1\*shape2. Can't prove it tho... but it works. \
 \
- Once the fix are ok and the algorithm finally works I'll have to optimize it. Optimization comes in two ways : use more integrated function (like comparing numpy array) and the way Pieces are stocked. Current way uses fully 3D matrix but I need to store the position of each block in 3D space. Will be MUCH faster to put pieces inside the map. \
- Upload the optimized minimalist version. Yeah because the original minimalist isn't very optimized.
- Fix the current bot version. \
 \
Maybe area : \
Better code \
Better implementation of pieces
