#include <iostream>
#include <vector>
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;
int main()
{
  vector<int > v{1,2,3};
  Vector2d v2(1,2);
  cout << v[1] <<endl;
  cout << "hello" << endl;
}