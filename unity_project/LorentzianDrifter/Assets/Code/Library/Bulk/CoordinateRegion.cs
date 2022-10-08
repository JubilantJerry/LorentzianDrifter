using System;
using UnityEngine;

public interface ICoordinateRegion
{
    bool ContainsPoint(Vector4 vec);
}

public class AndCoordinateRegion : ICoordinateRegion
{
    public AndCoordinateRegion(params ICoordinateRegion[] entries) =>
        _entries = entries;

    public bool ContainsPoint(Vector4 vec)
    {
        bool result = true;
        foreach (ICoordinateRegion entry in _entries)
        {
            result = result && entry.ContainsPoint(vec);
        }
        return result;
    }

    private ICoordinateRegion[] _entries;
}

public sealed class OrCoordinateRegion : ICoordinateRegion
{
    public OrCoordinateRegion(params ICoordinateRegion[] entries) =>
        _entries = entries;

    public bool ContainsPoint(Vector4 vec)
    {
        bool result = false;
        foreach (ICoordinateRegion entry in _entries)
        {
            result = result || entry.ContainsPoint(vec);
        }
        return result;
    }

    private ICoordinateRegion[] _entries;
}

public sealed class NotCoordinateRegion : ICoordinateRegion
{
    public NotCoordinateRegion(ICoordinateRegion arg) => _arg = arg;

    public bool ContainsPoint(Vector4 vec)
    {
        return !_arg.ContainsPoint(vec);
    }

    private ICoordinateRegion _arg;
}

public sealed class BoxCoordinateRegion : ICoordinateRegion
{
    public struct CoordinateRange
    {
        public CoordinateRange(int coord, float? min, float? max)
        {
            if (coord < 0 || coord >= 4)
            {
                throw new ArgumentOutOfRangeException(coord.ToString());
            }
            Coord = coord;
            Min = min;
            Max = max;
        }

        public int Coord { get; }
        public float? Min { get; }
        public float? Max { get; }
    }

    public BoxCoordinateRegion(Array4<CoordinateRange> ranges) =>
        Ranges = ranges;

    public bool ContainsPoint(Vector4 vec)
    {
        for (int i = 0; i < Ranges.Length; i++)
        {
            CoordinateRange coordRange = Ranges[i];
            if (coordRange.Min.HasValue &&
                vec[coordRange.Coord] < coordRange.Min.Value)
            {
                return false;
            }
            if (coordRange.Max.HasValue &&
                vec[coordRange.Coord] > coordRange.Max.Value)
            {
                return false;
            }
        }
        return true;
    }

    public Array4<CoordinateRange> Ranges { get; }
}

public sealed class DiskCoordinateRegion : ICoordinateRegion
{
    public struct CoordinateCenter
    {
        public CoordinateCenter(int coord, float center)
        {
            if (coord < 0 || coord >= 4)
            {
                throw new ArgumentOutOfRangeException(coord.ToString());
            }
            Coord = coord;
            Center = center;
        }

        public int Coord { get; }
        public float Center { get; }
    }

    public DiskCoordinateRegion(
        Array4<CoordinateCenter> center, float radius) =>
        (Center, Radius) = (center, radius);

    public bool ContainsPoint(Vector4 vec)
    {
        float dist = 0.0f;
        for (int i = 0; i < Center.Length; i++)
        {
            CoordinateCenter coordCenter = Center[i];
            float diff = vec[coordCenter.Coord] - coordCenter.Center;
            dist += diff * diff;
        }
        dist = Mathf.Sqrt(dist);
        return dist <= Radius;
    }

    public Array4<CoordinateCenter> Center { get; }
    public float Radius { get; }
}
