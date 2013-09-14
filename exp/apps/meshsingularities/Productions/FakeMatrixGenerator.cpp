/*
 * FakeMatrixGenerator.cpp
 *
 *  Created on: Aug 28, 2013
 *      Author: kjopek
 */

#include "FakeMatrixGenerator.h"

std::list<EquationSystem*> *FakeMatrixGenerator::CreateMatrixAndRhs(TaskDescription& taskDescription)
{

	std::list<EquationSystem*> *leafList = new std::list<EquationSystem*>();

	const int iSize = this->getiSize(taskDescription.polynomialDegree);
	const int leafSize = this->getLeafSize(taskDescription.polynomialDegree);
	const int a1Size = this->getA1Size(taskDescription.polynomialDegree);
	const int aNSize = this->getANSize(taskDescription.polynomialDegree);

	for (int i = 0; i < taskDescription.nrOfTiers; ++i) {
		int n;
		if (i==0) {
			n = a1Size;
		}
		else if (i==taskDescription.nrOfTiers-1) {
			n = aNSize;
		}
		else {
			n = leafSize + 2*iSize;
		}

		EquationSystem *system = new EquationSystem(n);
		for (int j=0; j<n; ++j) {
			system->matrix[j][j] = 1.0;
			system->rhs[j] = 1.0;
			system->matrix[j][0] = 2.0;
		}

		leafList->push_back(system);
	}

	return leafList;
}

void FakeMatrixGenerator::checkSolution(std::map<int,double> *solution_map, double (*f)(int dim, ...))
{
 // empty
}
