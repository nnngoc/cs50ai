import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines

class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if self.count == len(self.cells) and self.count != 0:
            return set(self.cells)
        
        return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return set(self.cells)
        
        return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            # Remove cell
            self.cells.remove(cell)
            # Update count
            self.count -= 1
        else: 
            return

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            # Remove cell
            self.cells.remove(cell)
        else: 
            return

class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    # Define function return neighbour of cell
    def neighbours(self, cell):
        # Create a set get all neighbor cells of current cell
        neighbours = set()
        
        for i in range(self.width):
            for j in range(self.height):
                if (i, j) in self.safes:
                    continue
                elif abs(i - cell[0]) >= 0 and abs(i - cell[0]) <= 1:
                    if abs(j - cell[1]) >= 0 and abs(j - cell[1]) <= 1:
                        neighbours.add((i, j))
                else: 
                    continue
        return neighbours

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        # Mark the cell as a move that has been made
        self.moves_made.add(cell)

        # Mark the cell as a safe cell
        self.mark_safe(cell)

        # Add a new sentence to the AI's knowledge base
        new_sentence = Sentence(self.neighbours(cell), count)
        if new_sentence not in self.knowledge:
            self.knowledge.append(new_sentence)

        # Mark new cells as safe or as mines
        for sentence in self.knowledge:
            known_safes = sentence.known_safes()
            for safe in known_safes:
                self.mark_safe(safe)
        
        for sentence in self.knowledge:    
            known_mines = sentence.known_mines()
            for mine in known_mines:
                self.mark_mine(mine)
        
        # Draw new inferences base on AI's knowledge
        new_kb = []
        for sen1 in self.knowledge:
            for sen2 in self.knowledge:
                # Skip if two sentences is the same
                if sen1 == sen2: 
                    continue
                # Skip if length of cells or count equal zero
                if sen1.count == 0 or sen2.count == 0 or len(sen1.cells) == 0 or len(sen2.cells) == 0:
                    continue
                # Draw new inferences
                if sen1.cells.issubset(sen2.cells):
                    cells = sen2.cells - sen1.cells
                    count = sen2.count - sen1.count
                    new_sen = Sentence(cells, count)
                    new_kb.append(new_sen)
        # Add new sentences to base knownledge
        for sen in new_kb:
            self.knowledge.append(sen)
        
    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        for cell in self.safes:
            if cell not in self.moves_made:
                print("Safe move: ", cell)
                return cell
        return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        # Total moves can have in game
        moves = self.height * self.width
        while moves > 0:
            
            # Random row, column move
            i = random.randrange(self.height)
            j = random.randrange(self.width)

            # Skip if move in moves made or mines
            if (i, j) not in self.moves_made | self.mines:
                return (i, j)

            moves -= 1
        return None
