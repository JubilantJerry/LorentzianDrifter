using System;
using System.Collections;
using System.Collections.Generic;

readonly public struct Array4<T> : IEnumerable<T>
{
    public Array4(T v0) : this(v0, default(T), default(T), default(T), 1) { }
    public Array4(T v0, T v1) : this(v0, v1, default(T), default(T), 2) { }
    public Array4(T v0, T v1, T v2) : this(v0, v1, v2, default(T), 3) { }
    public Array4(T v0, T v1, T v2, T v3) : this(v0, v1, v2, v3, 4) { }
    private Array4(T v0, T v1, T v2, T v3, int length) =>
        (_v0, _v1, _v2, _v3, Length) = (v0, v1, v2, v3, length);
    public Array4(T[] arr)
    {
        Length = arr.Length;
        switch (Length)
        {
            case 0:
                _v0 = default(T);
                _v1 = default(T);
                _v2 = default(T);
                _v3 = default(T);
                break;
            case 1:
                _v0 = arr[0];
                _v1 = default(T);
                _v2 = default(T);
                _v3 = default(T);
                break;
            case 2:
                _v0 = arr[0];
                _v1 = arr[1];
                _v2 = default(T);
                _v3 = default(T);
                break;
            case 3:
                _v0 = arr[0];
                _v1 = arr[1];
                _v2 = arr[2];
                _v3 = default(T);
                break;
            case 4:
                _v0 = arr[0];
                _v1 = arr[1];
                _v2 = arr[2];
                _v3 = arr[3];
                break;
            default:
                throw new System.ArgumentOutOfRangeException(
                    arr.Length.ToString());
        }
    }

    public T this[int i]
    {
        get
        {
            switch (i)
            {
                case 0: return _v0;
                case 1: return _v1;
                case 2: return _v2;
                case 3: return _v3;
                default: throw new IndexOutOfRangeException(i.ToString());
            }
        }
    }

    public IEnumerator<T> GetEnumerator()
    {
        switch (Length)
        {
            case 0:
                break;
            case 1:
                yield return _v0;
                break;
            case 2:
                yield return _v0;
                yield return _v1;
                break;
            case 3:
                yield return _v0;
                yield return _v1;
                yield return _v2;
                break;
            case 4:
                yield return _v0;
                yield return _v1;
                yield return _v2;
                yield return _v3;
                break;
            default:
                throw new InvalidOperationException(Length.ToString());
        }
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    private readonly T _v0, _v1, _v2, _v3;
    public int Length { get; }
}
