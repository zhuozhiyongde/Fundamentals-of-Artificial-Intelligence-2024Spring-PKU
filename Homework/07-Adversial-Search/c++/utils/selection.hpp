#pragma once

#include <vector>
#include <cassert>
#include <cfloat>

#include "random_variables.hpp"

// 选择算法基类，当done()为true时selected_index()即为选择的值序号
class SelectionBase{
public:

    SelectionBase() = default;
    virtual ~SelectionBase() = default;

    virtual void initialize(int items, double initial_value) = 0;
    virtual void submit(double value) = 0;
    virtual bool done() const = 0;
    virtual int selected_index() const = 0;
};

// 轮盘赌选择算法，选择某个值的概率与值本身成正比（要求值均为非负数）
class RouletteSelection : public SelectionBase{
private:

    double value_sum;
    int index, total_items, submitted_items;

public:

    RouletteSelection() = default;

    // items:备选集合大小，initial_value:与算法无关
    void initialize(int items, double initial_value) override {
        value_sum = 0;
        index = 0;
        total_items = items;
        submitted_items = 0;
    }

    void submit(double value) override {
        assert(value >= 0);
        value_sum += value;
        if (RandomVariables::uniform_real() < value / (value_sum + DBL_MIN)){
            index = submitted_items;
        }
        ++ submitted_items;
    }
    
    bool done() const override {
        return submitted_items >= total_items;
    }

    int selected_index() const override {
        return index;
    }
};

// 首个更优选择算法，选择第一个大于基准值的
class FirstBetterSelection : public SelectionBase{
private:

    double target_value;
    int index, total_items, submitted_items;
    bool item_found;

public:

    FirstBetterSelection() = default;
    
    // items:备选集合大小，initial_value:基准值
    void initialize(int items, double initial_value) override {
        target_value = initial_value;
        index = 0;
        total_items = items;
        submitted_items = 0;
        item_found = false;
    }

    void submit(double value) override {

        if (value > target_value){
            item_found = true;
            index = submitted_items;
        }

        ++ submitted_items;
    }

    bool done() const override {
        return item_found or submitted_items >= total_items;
    }

    int selected_index() const override {
        return index;
    }
};

// 最大选择算法，选择所有值中最大的
class MaxSelection : public SelectionBase{
private:

    double max_value;
    int index, total_items, submitted_items;

public:

    MaxSelection() = default;

    // items:备选集合大小，initial_value:与算法无关
    void initialize(int items, double initial_value) override {
        index = 0;
        total_items = items;
        submitted_items = 0;
    }

    void submit(double value) override {
        if (submitted_items == 0 or value > max_value){
            max_value = value;
            index = submitted_items;
        }
        ++ submitted_items;
    }

    bool done() const override {
        return submitted_items >= total_items;
    }

    int selected_index() const override {
        return index;
    }
};