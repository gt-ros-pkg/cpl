/************************************************************************
 *
 *  lap.h
   version 1.0 - 21 june 1996
   author  Roy Jonker, MagicLogic Optimization Inc.

   header file for LAP
   *
   **************************************************************************/
namespace cpl_visual_features
{

/*************** CONSTANTS  *******************/

#define BIG 100000

/*************** TYPES      *******************/

// TODO: Move into the namespace
typedef int LapRow;
typedef int LapCol;
typedef float LapCost;

/*************** FUNCTIONS  *******************/

extern LapCost lap(int dim, LapCost **assigncost,
                    LapCol *rowsol, LapRow *colsol, LapCost *u, LapCost *v);

extern void checklap(int dim, LapCost **assigncost,
                     LapCol *rowsol, LapRow *colsol, LapCost *u,
                     LapCost *v);
};
