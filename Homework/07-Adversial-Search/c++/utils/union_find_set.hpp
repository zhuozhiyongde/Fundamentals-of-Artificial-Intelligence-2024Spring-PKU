#pragma once

#include <vector>

// 并查集算法
class UnionFindSet{
private:
    
    // 保存元素所属的类别
    std::vector<int> color;

public:

    UnionFindSet(int n){
        for (int i = 0; i < n; ++ i){
            color.push_back(i);
        }
    }

    UnionFindSet() = default;

    // 查找一个元素所属的类别
    int find(int x){
        return x == color[x] ? x : (color[x] = find(color[x]));
    }

    // 合并两个元素所属的类别
    void join(int x, int y){
        int cx = find(x), cy = find(y);
        if (cx != cy){
            color[cx] = cy;
        }
    }
};