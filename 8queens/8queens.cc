#include <iostream>
#include <vector>
#include <array>
#include <sstream>
#include <random>
#include <algorithm>

using namespace std;

static const size_t SIZE_BOARD = 8;
static const size_t NUM_SQUARES = SIZE_BOARD * SIZE_BOARD;
static const size_t QUEENS = 8;

static const size_t EVOLUTIONS = 1000;
static const size_t POPULATION = 1000;

static std::random_device RD;
static std::mt19937 GEN(RD());
static std::uniform_int_distribution<> RANDSQR(0, NUM_SQUARES - 1);


enum State
{
   vacant = 0,
   threat,
   queen,
   queen_threat
};


class Board
{
private:
   array<State, NUM_SQUARES> squares;

protected:
   int pos (int row, int col)
   {
      return (SIZE_BOARD * row) + col;
   }

   void setThreat(int row, int col)
   {
      int t = pos(row, col);

      if (squares[t] == State::vacant)
         squares[t] = State::threat;
      else if (squares[t] == State::queen)
         squares[t] = State::queen_threat;
      /* other States, threat and queen_threat don't change */
   }

   void setThreats(int row, int col)
   {
      //set horizontal threats
      for (int c = 0; c < SIZE_BOARD; c++)
      {
         if (c != col)
            setThreat(row, c);
      }
      //set vertical threats
      for (int r = 0; r < SIZE_BOARD; r++)
      {
         if (r != row)
            setThreat(r, col);
      }
      //set diagonal threats
      //up-left
      for (int r = row, c = col; r >= 0 && c >= 0; r--, c--)
      {
         setThreat(r, c);
      }
      //up-right
      for (int r = row, c = col; r >= 0 && c < SIZE_BOARD; r--, c++)
      {
         setThreat(r, c);
      }
      //down-left
      for (int r = row, c = col; r < SIZE_BOARD && c >= 0; r++, c--)
      {
         setThreat(r, c);
      }
      //down-right
      for (int r = row, c = col; r < SIZE_BOARD && c < SIZE_BOARD; r++, c++)
      {
         setThreat(r, c);
      }
   }

public:
   //ctor
   Board()
   {
      for (int i = 0; i < squares.size(); i++)
      {
         squares[i] = State::vacant;
      }

      size_t placedQueens = 0;

      while (placedQueens < QUEENS)
      {
         int pos = RANDSQR(GEN);

         if (setQueen(pos))
         {
            placedQueens++;
         }
      }
   }

   //cross ctor
   Board(Board board1, Board board2)
   {
      //performance reasons
      if (board1.getSquares() == board2.getSquares())
      {
         squares = board1.getSquares();
         return;
      }

      //empty board
      for (int i = 0; i < squares.size(); i++)
      {
         squares[i] = State::vacant;
      }

      //mixing both boards
      int splitpos = RANDSQR(GEN);
      size_t placedQueens = 0;

      int p;
      for (p = 0; p < NUM_SQUARES && placedQueens < QUEENS; p++)
      {
         if (p < splitpos)
         {
            auto tmp = board1.getSquares().at(p);
            if ((tmp == State::queen) || (tmp == State::queen_threat))
            {
               if (setQueen(p))
                  placedQueens++;
            }
         }
         else
         {
            auto tmp = board2.getSquares().at(p);
            if ((tmp == State::queen) || (tmp == State::queen_threat))
            {
               if (setQueen(p))
                  placedQueens++;
            }
         }
      }

      //some queen is not set yet
      while (placedQueens < QUEENS)
      {
         int pos = RANDSQR(GEN);

         if (setQueen(pos))
         {
            placedQueens++;
         }
      }

   }

   array<State, NUM_SQUARES> getSquares() const
   {
      return squares;
   }

   bool setQueen(int position)
   {
      //inverse calculation to (row, col)
      int icol = position % SIZE_BOARD;
      int irow = (position - icol) / SIZE_BOARD;

      return setQueen(irow, icol);
   }

   bool setQueen(int row, int col)
   {
      int p = pos(row, col);

      if (squares[p] == State::vacant)
      {
         setThreats(row, col);
         squares[p] = State::queen;
         return true;
      }
      else if (squares[p] == State::threat)
      {
         setThreats(row, col);
         squares[p] = State::queen_threat;
         return true;
      }
      /* it cannot set queen if the row-col place is already occupied*/
      return false;
   }

   int getFitness() const
   {
      int queens_threat = std::count(squares.begin(), squares.end(), State::queen_threat);
      return QUEENS - queens_threat; /* Maximum fitness is all the QUEENS placed */
   }

   void mutate()
   {
      //move a queen_threat to a vacant square
      int pos_queen_threat = 0;
      int pos_vacant = 0;
      bool found = false;

      for (int i = 0; i < squares.size(); i++)
      {
         if (squares[i] == State::queen_threat)
            pos_queen_threat = i;
         else if ( (squares[i] == State::vacant) 
                  || (squares[i] == State::threat) )
            pos_vacant = i;

         //exit as soon as possible
         if ((pos_queen_threat != 0) && (pos_vacant != 0))
         {
            found = true;
            break;
         }
      }

      if (found)
      {
         squares[pos_queen_threat] = State::vacant;
         //try to set the queen in the vacant position
         if (!setQueen(pos_vacant))
            squares[pos_vacant] = State::queen_threat;
      }

   }

   string toString()
   {
      stringstream str;

      for (int r = 0; r < SIZE_BOARD; r++)
      {
         for (int c = 0; c < SIZE_BOARD; c++)
         {
            str << squares[pos(r,c)] << " ";
         }
         str << "\n";
      }

      return str.str();
   }

   void print()
   {
      cout << toString() << endl;
   }

};


class GA
{
   vector<Board> boards;
   const size_t population;
   const float elitism = 0.1f; //10%
   const float mutation = 0.25f; //25%
   
public:

   GA(size_t boardsNum = 100)
      :population(boardsNum)
   {
      for (size_t i = 0; i < boardsNum; i++)
      {
         boards.push_back(Board());
      }
      cout << "Boards generated." << endl;
   }

   Board getFittest()
   {
      return boards[0];
   }

   void selection()
   {
      //sort the current boards by fitness
      std::sort(boards.begin(), boards.end(), /*lambda comparison*/
                [](const Board& lhs, const Board& rhs)
                {
                   //first must be the greater fitness
                   return lhs.getFitness() > rhs.getFitness();
                }
               );
      
      //select only the elite
      size_t end = boards.size() * elitism;
      vector<Board> eliteBoards (boards.begin(), boards.begin() + end);
      boards = eliteBoards;
   }

   void crossover()
   {
      //populate the board till population size
      std::random_device rd;
      std::mt19937 gen(rd());
      //distribution range goes from 0 to current elite boards size
      std::uniform_int_distribution<> randBoard(0, boards.size()-1);
      //distribution for mutations
      std::uniform_int_distribution<> mutate(0,1);

      while (boards.size() < population)
      {
         Board board1 = boards[randBoard(gen)];
         Board board2 = boards[randBoard(gen)];
         Board child (board1, board2);

         //mutate?
         if (mutate(GEN) < mutation)
            child.mutate();
         
         boards.push_back(child);
      }
   }

   bool hasSolution()
   {
      return boards[0].getFitness() == QUEENS;
   }

};


/************/
int main()
{
   GA ga(POPULATION);
   
   int evo;
   for (evo = 0; evo < EVOLUTIONS; evo++)
   {
      if (ga.hasSolution()) break;
      
      ga.selection();
      ga.crossover();
   }

   cout << "Solution board:\n" << ga.getFittest().toString() << endl;
   cout << "Fitness (no killed queens): " << ga.getFittest().getFitness() << endl;
   cout << "Evolution: " << evo << endl;

   return 0;
}
