import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Error():
    def __init__(self, msg):
        print(msg)

class all_piece():
    def __init__(self, shape=None):
        self.shape = shape
        self.piece_list = []
        self.piece_dim = []
        self.piece_size = []
        self.piece_number = []
        self.total_piece = 0
        self.forbidden_places = []
        self.val_to_id = []
        self.val_to_number = []
        self.solution_list = []
        self.solution_rotsym_neighbour = []
        self.solution_rotsym_interact = []
        self.total_solution = 0
        self.color_list = ["#FF0000CC", "#00FF00CC", "#0000FFCC", "#FF00FFCC", "#FFFF00CC", "#00FFFFCC", "#000000CC", "#FFFFFFCC", "#888888CC", "#444444CC", "#BBBBBBCC"]
        self.reset_neighbour = []
        self.reset_interact = []
        self.sum_list = [0,1,3,6,10,15,21]
        self.prime_list = []

    def rotsym_neighbour(self, p_map):
        rotsym = 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    if p_map[i,j,k] != -1:
                        neighbour = copy.copy(self.reset_neighbour)
                        if i > 0 and p_map[i][j][k] != p_map[i-1][j][k]:
                            neighbour[self.val_to_id[ int(p_map[i-1][j][k]) ]] += 1
                        if j > 0 and p_map[i][j][k] != p_map[i][j-1][k]:
                            neighbour[self.val_to_id[ int(p_map[i][j-1][k]) ]] += 1
                        if k > 0 and p_map[i][j][k] != p_map[i][j][k-1]:
                            neighbour[self.val_to_id[ int(p_map[i][j][k-1]) ]] += 1
                        if i < self.shape[0]-1 and p_map[i][j][k] != p_map[i+1][j][k]:
                            neighbour[self.val_to_id[ int(p_map[i+1][j][k]) ]] += 1
                        if j < self.shape[1]-1 and p_map[i][j][k] != p_map[i][j+1][k]:
                            neighbour[self.val_to_id[ int(p_map[i][j+1][k]) ]] += 1
                        if k < self.shape[2]-1 and p_map[i][j][k] != p_map[i][j][k+1]:
                            neighbour[self.val_to_id[ int(p_map[i][j][k+1]) ]] += 1

                        sum = 1
                        for x in neighbour:
                            sum *= self.sum_list[x+1]
                        rotsym += sum
        return rotsym

    def rotsym_interact(self, p_map):
        interaction = copy.copy(self.reset_interact)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    if i > 0:
                        interaction[int(p_map[i][j][k]-1), int(p_map[i-1][j][k]-1)] += 1
                    if j > 0:
                        interaction[int(p_map[i][j][k]-1), int(p_map[i][j-1][k]-1)] += 1
                    if k > 0:
                        interaction[int(p_map[i][j][k]-1), int(p_map[i][j][k-1]-1)] += 1

        rotsym = 1
        for i in range(len(interaction)):
            for j in range(i+1, len(interaction[0])):
                rotsym *= self.prime_list[int(interaction[i,j]) + int(interaction[j,i]) + 1]
        return rotsym

    def create_minicube(self, piece_map):
        max_piece_dim = max(self.get_dim(piece_map))+1
        minicube = np.zeros([max_piece_dim]*3)
        for i in range(min(piece_map.shape[0], minicube.shape[0])):
            for j in range(min(piece_map.shape[1], minicube.shape[1])):
                for k in range(min(piece_map.shape[2], minicube.shape[2])):
                    minicube[i,j,k] = piece_map[i,j,k]
        return minicube

    def print_3D(self, piece_map, view_3D=False):
        #if (view_3D == 1):
        #    get_ipython().magic('matplotlib notebook')
        #else:
        #    get_ipython().magic('matplotlib inline')
        piece_map = self.create_minicube(piece_map)
        x, y, z = np.indices(piece_map.shape)
        voxels = (piece_map != 0)
        #print(voxels)
        colors = np.empty(voxels.shape, dtype=object)
        for i in range(piece_map.shape[0]):
            for j in range(piece_map.shape[1]):
                for k in range(piece_map.shape[2]):
                    value = int(piece_map[i,j,k])
                    if value > 0:
                        colors[i,j,k] = self.color_list[value - 1]

        colors = self.rotate(colors, "l")
        voxels = self.rotate(voxels, "l")
        fig = plt.figure(figsize=(2,2))
        ax = fig.gca(projection='3d')
        ax.voxels(voxels, facecolors=colors, edgecolor='k')
        plt.show()

    def is_same_piece(self, piece, cmpr):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    if (piece[i,j,k] == 0 and cmpr[i,j,k] != 0) or (piece[i,j,k] != 0 and cmpr[i,j,k] == 0):
                        return 0
        return 1

    def is_exist(self, piece, piece_list):
        for i in piece_list:
            if self.is_same_piece(piece, i) == 1:
                return 1
        return 0

    def is_fitting(self, piece_dim):
        for i in range(3):
            if piece_dim[i] >= self.shape[i]:
                return 0
        return 1

    def can_fit(self, piece_dim):
        for i,j in zip(sorted(piece_dim),sorted(self.shape)):
            if i >= j:
                return 0
        return 1

    def get_dim(self, piece):
        dim = [0,0,0]
        for i in range(piece.shape[0]):
            for j in range(piece.shape[1]):
                for k in range(piece.shape[2]):
                    if piece[i,j,k] != 0:
                        if dim[0] < i:
                            dim[0] = i
                        if dim[1] < j:
                            dim[1] = j
                        if dim[2] < k:
                            dim[2] = k
        return dim

    def get_size(self, piece):
        size = 0
        for i in range(piece.shape[0]):
            for j in range(piece.shape[1]):
                for k in range(piece.shape[2]):
                    size += (piece[i,j,k] != 0)
        return size

    def get_piece_map(self, piece_id, rotation=0):
        return self.piece_list[piece_id][rotation]

    def goto_selfshape(self, piece, piece_dim):
        new_array = np.zeros(self.shape)
        for i in range(piece_dim[0]+1):
            for j in range(piece_dim[1]+1):
                for k in range(piece_dim[2]+1):
                    new_array[i,j,k] = piece[i,j,k]
        return new_array

    def replace_piece(self, piece):
        fix_pos = []
        min_pos = np.array(piece.shape)
        for i in range(piece.shape[0]):
            for j in range(piece.shape[1]):
                for k in range(piece.shape[2]):
                    if piece[i,j,k] != 0:
                        piece[i,j,k] = 0
                        fix_pos.append([i,j,k])
                        if min_pos[0] > i:
                            min_pos[0] = i
                        if min_pos[1] > j:
                            min_pos[1] = j
                        if min_pos[2] > k:
                            min_pos[2] = k
        for i in fix_pos:
            piece[i[0] - min_pos[0], i[1] - min_pos[1], i[2] - min_pos[2]] = 1
        return piece

    def rotate(self, piece, rotate_code):
        for i in rotate_code:
            if i == "r":
                piece = np.rot90(piece,  1, (0,1))
            if i == "l":
                piece = np.rot90(piece, -1, (0,1))
            if i == "b":
                piece = np.rot90(piece,  1, (0,2))
            if i == "f":
                piece = np.rot90(piece, -1, (0,2))
            if i == "d":
                piece = np.rot90(piece,  1, (1,2))
            if i == "u":
                piece = np.rot90(piece, -1, (1,2))
        return np.copy(piece)

    def gen_piece(self, piece, gen_type=None):
        piece_list = []
        self.piece_dim.append([])
        self.piece_size.append(self.get_size(piece))
        #map_cost_ref = get_map_cost_3D(piece)
        #print("map_cost ref :", map_cost_ref)
        code_faces = "fulful"
        code_490rot = "ulfulf"
        for i in range(6):
            for j in range(4):
                piece = self.replace_piece(piece)
                piece_dim = self.get_dim(piece)
                if self.is_fitting(piece_dim):
                    if piece.shape != self.shape:
                        piece = self.goto_selfshape(piece, piece_dim)
                    if self.is_exist(piece, piece_list) == 0:
                        piece_list.append(piece)
                        self.piece_dim[-1].append(piece_dim)
                        if gen_type == "fixed":
                            return piece_list
                        #print(piece_dim)
                piece = self.rotate(np.copy(piece), code_490rot[i])
            piece = self.rotate(np.copy(piece), code_faces[i])
        return piece_list

    def to_numpy(self, piece): #definitely need to improve this fnc
        if type(piece) != type(np.array([])):
            piece = np.asarray(piece)
        if len(piece.shape) > 3:
            raise Error("Dimension of piece is > 3")
        res = np.zeros(self.shape)
        if len(piece.shape) == 0:
            res[0,0,0] = 1
            return res
        res = np.zeros([max(self.shape)]*3)
        cursor = [0,0,0]
        for idx, i in enumerate(piece):
            cursor[0] = idx
            try:
                i[0]
            except (TypeError, IndexError):
                res[cursor[0], cursor[1], cursor[2]] = i
            else:
                for jdx, j in enumerate(i):
                    cursor[1] = jdx
                    try:
                        j[0]
                    except (TypeError, IndexError):
                        res[cursor[0], cursor[1], cursor[2]] = j
                    else:
                        for kdx, k in enumerate(j):
                            cursor[2] = kdx
                            res[cursor[0], cursor[1], cursor[2]] = k
        res[res != 0] = 1
        return res

    def add_piece(self, piece, number=1, add_type=None):
        if self.shape == None:
            raise Error('Define the shape of the puzzle first')
        if type(number) != type(0):
            raise Error('number of piece is not int')
        if number < 1:
            raise Error('number cannot be less than 1')
        piece = self.to_numpy(piece)
        #print(piece)
        if add_type == "fixed":
            if self.is_fitting(self.get_dim(piece)) == 0:
                print("Current dimension :",self.get_dim(piece))
                raise Error("The piece is not fitting, change piece or puzzle's shape\n")
        if self.can_fit(self.get_dim(piece)) == 0:
            raise Error("The piece is not fitting, change piece or puzzle's shape")
        self.piece_list.append(self.gen_piece(piece, add_type))
        self.piece_number.append(number)

    def print_all(self, piece_id=None):
        if piece_id != None:
            print("piece", piece_id, " size :", self.piece_size[piece_id], " rotation :", len(self.piece_list[piece_id]), " copies :", self.piece_number[0])
            for idx, i in enumerate(self.piece_list[piece_id]):
                #print(i)
                self.print_3D(i)
            print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        else:
            for idx, i in enumerate(self.piece_list):
                print("piece", idx, " size :", self.piece_size[idx], " rotation :", len(self.piece_list[idx]), " copies :", self.piece_number[idx])
                #print(i[0])
                self.print_3D(i[0])
                print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
                #print("dim :", self.piece_dim[idx][0])

    def watch_solution(self, display_type=None):
        print("Number of solutions :", self.total_solution)
        if display_type != None:
            for i in self.solution_list:
                if display_type == "3D":
                    self.print_3D(i)
                else:
                    print(i)

    def is_solvable(self):
        space = self.shape[0] * self.shape[1] * self.shape[2]
        piece_space = 0
        for i in range(len(self.piece_list)):
            piece_space += self.piece_size[i] * self.piece_number[i]
        if piece_space == space:
            return 1
        if piece_space < space:
            print("The amount of space taken by pieces is too low to fill the puzzle")
            print("Change and/or add pieces")
        else:
            print("The amount of space taken by pieces is too high to fill the puzzle")
            print("Change and/or remove pieces")
        print("Space to fill :", space, "  Space of all pieces :", piece_space)
        return 0

    def is_placeable(self, pos, piece_map, piece):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    if piece[i,j,k] != 0 and piece_map[i+pos[0], j+pos[1], k+pos[2]] != 0:
                        return 0
        return 1

    def is_allowed(self, i,j,k, piece_id, piece_numeral, piece_rotation):
        #print(piece_numeral)
        if self.piece_number[piece_id] == 1:
            return 1
        val = self.forbidden_places[piece_id][piece_numeral][piece_rotation][i,j,k]
        #print(val, piece_rotation_id)
        if val == 0:
            return 0
        for number in range(len(self.forbidden_places[piece_id])):
            self.forbidden_places[piece_id][number][piece_rotation][i,j,k] = 0
        self.forbidden_places[piece_id][piece_numeral][piece_rotation][i,j,k] = 1
        return 1

    def place_piece(self, pos, piece_map, piece, piece_value): #opti by saving piece bloc position plz
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    if piece[i,j,k] != 0:
                        piece_map[i+pos[0], j+pos[1], k+pos[2]] = piece_value
        return piece_map

    def recursive(self, piece_map, depth=0):
        if depth == self.total_piece:
            rotsym_i = self.rotsym_interact(piece_map)
            if rotsym_i in self.solution_rotsym_interact:
                return
            self.solution_rotsym_interact.append(rotsym_i)
            rotsym_n = self.rotsym_neighbour(piece_map)
            if rotsym_n in self.solution_rotsym_neighbour:
                return
            self.solution_rotsym_neighbour.append(rotsym_n)
            self.solution_list.append(piece_map)
            self.total_solution += 1
            print(self.total_solution, rotsym_n, rotsym_i, is_already_in)
            #print(piece_map)
            return
        piece_id = self.val_to_id[depth]
        piece_numeral = self.val_to_number[depth]
        for idx, piece_rotation in enumerate(self.piece_list[piece_id]):
            for i in range(self.shape[0] - self.piece_dim[piece_id][idx][0]):
                for j in range(self.shape[1] - self.piece_dim[piece_id][idx][1]):
                    for k in range(self.shape[2] - self.piece_dim[piece_id][idx][2]):
                        #print(depth, idx, i, j, k)
                        if self.is_placeable((i,j,k), piece_map, piece_rotation) == 1 \
                        and self.is_allowed(i,j,k, piece_id, piece_numeral, idx) == 1:
                            self.recursive(self.place_piece((i,j,k), np.copy(piece_map), piece_rotation, depth + 1), depth + 1)

    def init_forbidden(self):
        self.forbidden_places = []
        for idx, piece in enumerate(self.piece_list):
            if self.piece_number[idx] == 1:
                self.forbidden_places.append([1])
            else:
                self.forbidden_places.append([])
                for number in range(self.piece_number[idx]):
                    self.forbidden_places[idx].append([])
                    for rotation in range(len(self.piece_list[idx])):
                        self.forbidden_places[idx][number].append(np.ones(self.shape))
                    #print(self.forbidden_places[idx][number])

    def init_prime_list(self):
        prime_list = [0,1]
        ceil_value = 0
        print(self.piece_size)
        for i in range(len(self.piece_size)):
            print(self.piece_size[i], self.piece_number[i])
            tmp = self.piece_size[i] * self.piece_number[i] * 6
            if ceil_value < tmp:
                ceil_value = tmp
        prime_total = 0
        prime_current = 2
        while prime_total < ceil_value:
            is_prime = 1
            for i in prime_list[2:]:
                if prime_current % i == 0:
                    is_prime = 0
                    break
            if is_prime == 1:
                prime_total += 1
                prime_list.append(prime_current)
            prime_current += 1
        return prime_list

    def solve(self, solve_type=None):
        if self.is_solvable() == 0:
            return

        self.total_piece = sum(self.piece_number)
        self.val_to_id = []
        self.val_to_number = [0]
        for idx, i in enumerate(self.piece_number):
            for j in range(i):
                self.val_to_id.append(idx)
        self.val_to_id.append(len(self.piece_list))

        for i in range(1, len(self.val_to_id)):
            if self.val_to_id[i-1] == self.val_to_id[i]:
                self.val_to_number.append(self.val_to_number[i-1] + 1)
            else:
                self.val_to_number.append(0)

        print("depth to id >", self.val_to_id)
        print("depth to number >", self.val_to_number)
        self.solution_rotsym_neighbour = []
        self.solution_rotsym_interact = []
        self.solution_list = []
        self.total_solution = 0
        self.init_forbidden()
        self.reset_neighbour = [0] * (len(self.piece_list) + 1)
        self.reset_interact = np.zeros((self.total_piece, self.total_piece))
        self.prime_list = self.init_prime_list()

        self.recursive(np.zeros(self.shape))

piece_tL  = [[1,1],[1]]
piece_sqr = [[1,1],[1,1]]
piece_I2  = [1,1]
piece_1 = 1
piece_T = [[1,1,1],[0,1,0]]
piece_Z = [[1,1],[0,1,1]]
piece_L = [[1,1,1],[1]]
dtb = all_piece((3,3,3))

#dtb.add_piece(piece_L,4)
dtb.add_piece(piece_L,4)
dtb.add_piece(piece_Z)
dtb.add_piece(piece_T,1)
dtb.add_piece(piece_tL,1)

#dtb.print_3D([[[2,2],[1,1]],[[3,3],[3,1]]], True)
#dtb.print_all()


dtb.solve("yeye")
#dtb.watch_solution("3D")
