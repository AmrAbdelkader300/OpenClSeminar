
__kernel void HelloWorld(__global char* data)
{
    data[0] = 'H';
    data[1] = 'E';
    data[2] = 'L';
    data[3] = 'L';
    data[4] = 'O';
    data[5] = ' ';
    data[6] = 'W';
    data[7] = 'O';
    data[8] = 'R';
    data[9] = 'L';
    data[10] = 'D';
    data[11] = '!';
    data[12] = '\n';
}

__kernel void HelloEveryOne(__global char* data)
{
    data[0] = 'H';
    data[1] = 'E';
    data[2] = 'L';
    data[3] = 'L';
    data[4] = 'O';
    data[5] = ' ';
    data[6] = 'E';
    data[7] = 'V';
    data[8] = 'E';
    data[9] = 'R';
    data[10] = 'Y';
    data[11] = 'O';
    data[12] = 'N';
    data[13] = 'E';
    data[14] = '!';
    data[15] = '\n';
}