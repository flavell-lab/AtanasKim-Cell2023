# bin_img()
test_array1 = [1 0 1 0;
0 0 0 0;
1 0 1 0;
0 0 0 0]

test_array2 = [0 1 0 1;
0 0 0 0;
0 1 0 1;
0 0 0 0]

test_array3 = [0 0 0 0;
1 0 1 0;
0 0 0 0;
1 0 1 0]

test_array4 = [0 0 0 0;
0 1 0 1;
0 0 0 0;
0 1 0 1]

@test [0.25 0.25; 0.25 0.25] == ND2Process.bin_img(test_array1)
@test [0.25 0.25; 0.25 0.25] == ND2Process.bin_img(test_array2)
@test [0.25 0.25; 0.25 0.25] == ND2Process.bin_img(test_array3)
@test [0.25 0.25; 0.25 0.25] == ND2Process.bin_img(test_array4)
