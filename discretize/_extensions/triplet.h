#ifndef __TRIPLET_H
#define __TRIPLET_H

#include <cstddef>
#include <utility>
#include <functional>

template<class T, class U, class V>
struct triplet{
    T v1;
    U v2;
    V v3;

    triplet(){
        v1 = NULL;
        v2 = NULL;
        v3 = NULL;
    }

    triplet(T first, U second, V third){
        v1 = first;
        v2 = second;
        v3 = third;
    }
    bool operator==(const triplet<T, U, V> &other) const{
        return (v1 == other.v1
            && v2 == other.v2
            && v3 == other.v3);
    }
    bool operator<(const triplet<T, U, V> &other) const{
        return ! (
            (v1 > other.v1)
            || (v2 > other.v2)
            || (v3 > other.v3)
        );
    }
};

namespace std {

  template<class T, class U, class V>
  struct hash<triplet<T, U, V> >
  {
    std::size_t operator()(const triplet<T, U, V>& k) const
    {
      using std::hash;
      // Compute individual hash values for first,
      // second and third and combine them using XOR
      // and bit shifting:

      return ((hash<T>()(k.v1)
               ^ (hash<U>()(k.v2) << 1)) >> 1)
               ^ (hash<V>()(k.v3) << 1);
    }
  };

  template<class T, class U>
  struct hash<pair<T, U> >
  {
    size_t operator()(const pair<T, U>& k) const
    {
      using std::hash;

      return ((hash<T>()(k.first)
               ^ (hash<U>()(k.second) << 1)) >> 1);
    }
  };
}
#endif
