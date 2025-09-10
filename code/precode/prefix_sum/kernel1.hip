#include <iostream>
#include <vector>
using namespace std;

int main() {
    int n;
    cin >> n;  
    vector<int> arr(n);
    for (int i = 0; i < n; ++i) {
        cin >> arr[i];  
    }
    vector<int> prefixSum(n);
    prefixSum[0] = arr[0];  
    for (int i = 1; i < n; ++i) {
        prefixSum[i] = prefixSum[i - 1] + arr[i];
    }
    for (int i = 0; i < n; ++i) {
        cout << prefixSum[i] << " ";
    }
    cout << endl;

    return 0;
}
