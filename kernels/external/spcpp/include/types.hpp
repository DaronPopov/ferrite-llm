#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <iostream>

namespace spc {

template<typename T>
class List : public std::vector<T> {
public:
    using std::vector<T>::vector;

    void append(const T& val) { this->push_back(val); }
    
    bool contains(const T& val) const {
        return std::find(this->begin(), this->end(), val) != this->end();
    }

    void extend(const List<T>& other) {
        this->insert(this->end(), other.begin(), other.end());
    }

    void print() const {
        std::cout << "[";
        for (size_t i = 0; i < this->size(); ++i) {
            std::cout << (*this)[i] << (i == this->size() - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;
    }
};

template<typename K, typename V>
class Dict : public std::unordered_map<K, V> {
public:
    using std::unordered_map<K, V>::unordered_map;

    bool has(const K& key) const {
        return this->find(key) != this->end();
    }

    List<K> keys() const {
        List<K> k;
        for (auto const& [key, val] : *this) k.append(key);
        return k;
    }

    void print() const {
        std::cout << "{";
        size_t i = 0;
        for (auto const& [key, val] : *this) {
            std::cout << key << ": " << val << (i == this->size() - 1 ? "" : ", ");
            i++;
        }
        std::cout << "}" << std::endl;
    }
};

using String = std::string;

} // namespace spc
