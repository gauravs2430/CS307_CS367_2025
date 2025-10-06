import numpy as np
from PIL import Image
import random
import math

class PuzzlePiece:
    def __init__(self, image, position):
        self.image = image
        self.position = position

class JigsawPuzzle:
    def __init__(self, image_path, pieces_x, pieces_y):
        self.original_image = Image.open(image_path).convert("L")  # Convert to grayscale
        self.pieces_x = pieces_x
        self.pieces_y = pieces_y
        self.pieces = self.split_image()
        self.current_solution = self.pieces.copy()
        random.shuffle(self.current_solution)

    def split_image(self):
        width, height = self.original_image.size
        piece_width = width // self.pieces_x
        piece_height = height // self.pieces_y
        pieces = []
        for y in range(self.pieces_y):
            for x in range(self.pieces_x):
                left = x * piece_width
                upper = y * piece_height
                right = left + piece_width
                lower = upper + piece_height
                piece = self.original_image.crop((left, upper, right, lower))
                pieces.append(PuzzlePiece(piece, (x, y)))
                
                
        return pieces

    def calculate_cost(self, solution):
        cost = 0
        for i, piece in enumerate(solution):
            x, y = i % self.pieces_x, i // self.pieces_x
            if x > 0:
                left_piece = solution[i - 1]
                cost += self.edge_difference(piece.image, left_piece.image, 'left')
            if y > 0:
                top_piece = solution[i - self.pieces_x]
                cost += self.edge_difference(piece.image, top_piece.image, 'top')
        return cost

    def edge_difference(self, piece1, piece2, direction):
        
        if direction == 'left':
            edge1 = np.array(piece1)[:, 0]
            edge2 = np.array(piece2)[:, -1]
        else:  
            edge1 = np.array(piece1)[0, :]
            edge2 = np.array(piece2)[-1, :]
    
        
        mean1, mean2 = np.mean(edge1), np.mean(edge2)
        numerator = np.sum((edge1 - mean1) * (edge2 - mean2))
        denominator = np.sqrt(np.sum((edge1 - mean1) ** 2) * np.sum((edge2 - mean2) ** 2))
        correlation = numerator / denominator if denominator != 0 else 0
    
        
        return 1 - correlation


    def swap_pieces(self, solution, i, j):
        
        solution[i], solution[j] = solution[j], solution[i]

    def adaptive_simulated_annealing(self, initial_temp, min_temp, cooling_factor, num_iterations):
        current_solution = self.current_solution.copy()
        current_cost = self.calculate_cost(current_solution)
        best_solution = current_solution.copy()
        best_cost = current_cost
        temp = initial_temp
        stagnant_iterations = 0
        max_stagnant_iterations = 1000

        for iteration in range(num_iterations):
            
            i, j = random.sample(range(len(current_solution)), 2)
            self.swap_pieces(current_solution, i, j)
            new_cost = self.calculate_cost(current_solution)
            delta_cost = new_cost - current_cost
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / temp):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost
                    stagnant_iterations = 0
                else:
                    stagnant_iterations += 1
            else:
                self.swap_pieces(current_solution, i, j)  
                stagnant_iterations += 1
            # Adaptive cooling and reheating
            if stagnant_iterations > max_stagnant_iterations:
                temp = min(temp * 2, initial_temp)  # Reheat
                stagnant_iterations = 0
                print(f"Reheating at iteration {iteration}, New temp: {temp}")
            else:
                temp = max(temp * cooling_factor, min_temp)

            # Print progress every 1000 iterations
            if iteration % 1000 == 0:
                print(f"Iteration {iteration}, Current Cost: {current_cost}, Best Cost: {best_cost}, Temp: {temp}")

        self.current_solution = best_solution
        
        return best_solution, best_cost

    def solve(self, initial_temp=100, min_temp=0.1, cooling_factor=0.99, num_iterations=400000):
        
        return self.adaptive_simulated_annealing(initial_temp, min_temp, cooling_factor, num_iterations)

    def display_solution(self, solution):
        
        width, height = self.original_image.size
        
        piece_width = width // self.pieces_x
        
        piece_height = height // self.pieces_y
        result_image = Image.new('L', (width, height))
        for i, piece in enumerate(solution):
            x, y = i % self.pieces_x, i // self.pieces_x
            result_image.paste(piece.image, (x * piece_width, y * piece_height))
            
        result_image.show()
        result_image.save("solved_puzzle_grayscale_1.png")

# Usage example

if __name__ == "__main__":
    puzzle = JigsawPuzzle("input_image.png", 4, 4)  # Assuming a 4x4 puzzle
    best_solution, best_cost = puzzle.solve()
    print(f"Best solution cost: {best_cost}")
    puzzle.display_solution(best_solution)