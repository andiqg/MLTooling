using MLTooling.Core;
using MLTooling.Compute;

Shape shape = new Shape(3, 3);
using Tensor tensor = new Tensor(shape);

tensor[1,1] = 3f;
float one = tensor[1,1];
float two = tensor[0,0];

float[] buffer = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

using Tensor tensor2 = new Tensor(buffer, shape);

for (ulong i = 0; i < 3; i++)
{
    for (ulong j = 0; j < 3; j++)
        Console.WriteLine(tensor2[i, j]);
}

Console.WriteLine(one);
Console.WriteLine(two);


Console.WriteLine("========= Shape Test =============");
Shape testShape = new Shape(1);
Console.WriteLine(testShape.ToString());
