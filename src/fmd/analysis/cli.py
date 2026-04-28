"""Command-line interface entry points."""

import argparse
import sys
from datetime import datetime

import pandas as pd

from fmd.analysis.loaders import load_file
from fmd.analysis.plots import plot_time_series
from fmd.analysis.generate import generate_random_data as _generate_random_data
from fmd.analysis.processing import analyze_sampling as _analyze_sampling


def analyze_log():
    """CLI entry point for analyzing CSV data and plotting numeric time series."""
    parser = argparse.ArgumentParser(description="Analyze data from CSV (schema auto-detected)")
    parser.add_argument("filename", help="Input CSV filename")
    parser.add_argument(
        "--schema",
        default=None,
        help="Schema name (default: auto-detect). Try: test_data, vakaros_atlas_telemetry",
    )

    args = parser.parse_args()

    try:
        result = load_file(args.filename, schema=args.schema)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)

    df = result.df
    print(f"Loaded {len(df)} rows from {args.filename}")
    print(f"Schema: {result.schema_name}\n")
    print(df)

    plot_time_series(df)


def generate_data():
    """CLI entry point for generating random data."""
    parser = argparse.ArgumentParser(description="Generate random time series data")
    parser.add_argument("filename", help="Output CSV filename")
    parser.add_argument("-n", "--num-streams", type=int, default=3, help="Number of data streams")
    parser.add_argument("-s", "--time-step", type=float, default=0.1, help="Time step between samples")
    parser.add_argument("-e", "--end-time", type=float, default=1.0, help="End time")

    args = parser.parse_args()
    _generate_random_data(args.filename, args.num_streams, args.time_step, args.end_time)


def _parse_time_of_day(time_str: str) -> datetime.time:
    """Parse time-of-day string in various formats (HH:MM, HH:MM:SS, etc.)."""
    formats = [
        "%H:%M:%S",      # 13:30:00
        "%H:%M",         # 13:30
        "%I:%M:%S %p",   # 1:30:00 PM
        "%I:%M %p",      # 1:30 PM
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(time_str.strip(), fmt)
            return dt.time()
        except ValueError:
            continue

    raise ValueError(
        f"Could not parse time '{time_str}'. "
        f"Expected formats: HH:MM, HH:MM:SS, H:MM AM/PM, H:MM:SS AM/PM"
    )


def _time_of_day_to_seconds(
    time_of_day: datetime.time,
    first_timestamp: pd.Timestamp,
) -> float:
    """Convert time-of-day to seconds from the start of recording."""
    # Create datetime using the date from first timestamp and the time-of-day
    target_datetime = pd.Timestamp.combine(first_timestamp.date(), time_of_day)

    # Preserve timezone if the first timestamp is timezone-aware
    if first_timestamp.tz is not None:
        target_datetime = target_datetime.tz_localize(first_timestamp.tz)

    # If time-of-day is before the first timestamp's time, assume next day
    if target_datetime < first_timestamp:
        target_datetime += pd.Timedelta(days=1)

    # Convert to seconds from start
    return (target_datetime - first_timestamp).total_seconds()


def plot_vakaros():
    """CLI entry point for plotting Vakaros Atlas telemetry data with optional time range filtering."""
    parser = argparse.ArgumentParser(
        description="Load and plot Vakaros Atlas telemetry data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot entire file
  plot-vakaros data.csv

  # Plot using seconds from start
  plot-vakaros data.csv --start-time 50 --end-time 200

  # Plot using time of day (local timezone)
  plot-vakaros data.csv --start-time-of-day "13:30:00" --end-time-of-day "14:00:00"
  plot-vakaros data.csv --start-time-of-day "1:30 PM" --end-time-of-day "2:00 PM"
        """
    )
    parser.add_argument("filename", help="Vakaros Atlas CSV filename")

    # Time range in seconds
    time_group = parser.add_argument_group("time range (seconds from start)")
    time_group.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Start time in seconds from start of recording",
    )
    time_group.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="End time in seconds from start of recording",
    )

    # Time range as time of day
    time_of_day_group = parser.add_argument_group("time range (time of day)")
    time_of_day_group.add_argument(
        "--start-time-of-day",
        type=str,
        default=None,
        help="Start time as time of day (e.g., '13:30:00' or '1:30 PM')",
    )
    time_of_day_group.add_argument(
        "--end-time-of-day",
        type=str,
        default=None,
        help="End time as time of day (e.g., '14:00:00' or '2:00 PM')",
    )

    args = parser.parse_args()

    # Validate that we're not mixing time formats
    if (args.start_time is not None or args.end_time is not None) and \
       (args.start_time_of_day is not None or args.end_time_of_day is not None):
        print("Error: Cannot mix --start-time/--end-time with --start-time-of-day/--end-time-of-day", file=sys.stderr)
        raise SystemExit(2)

    # Load the file
    try:
        result = load_file(args.filename, schema="vakaros_atlas_telemetry")
    except ValueError as e:
        print(f"Error loading file: {e}", file=sys.stderr)
        raise SystemExit(2)

    df = result.df

    # Convert time-of-day to seconds if specified
    start_time_seconds = args.start_time
    end_time_seconds = args.end_time

    if args.start_time_of_day is not None or args.end_time_of_day is not None:
        # Need to read original CSV to get timestamp column
        try:
            raw_df = pd.read_csv(args.filename)
            if "timestamp" not in raw_df.columns:
                print("Error: File does not contain 'timestamp' column", file=sys.stderr)
                raise SystemExit(2)

            first_timestamp = pd.to_datetime(raw_df["timestamp"].iloc[0])

            if args.start_time_of_day is not None:
                time_of_day = _parse_time_of_day(args.start_time_of_day)
                start_time_seconds = _time_of_day_to_seconds(time_of_day, first_timestamp)

            if args.end_time_of_day is not None:
                time_of_day = _parse_time_of_day(args.end_time_of_day)
                end_time_seconds = _time_of_day_to_seconds(time_of_day, first_timestamp)

        except Exception as e:
            print(f"Error parsing time-of-day: {e}", file=sys.stderr)
            raise SystemExit(2)

    # Filter by time range if specified
    if start_time_seconds is not None or end_time_seconds is not None:
        mask = pd.Series(True, index=df.index)

        if start_time_seconds is not None:
            mask &= df["time"] >= start_time_seconds
            if not mask.any():
                print(f"Warning: No data found after start_time={start_time_seconds:.1f}s", file=sys.stderr)

        if end_time_seconds is not None:
            mask &= df["time"] <= end_time_seconds
            if not mask.any():
                print(f"Warning: No data found before end_time={end_time_seconds:.1f}s", file=sys.stderr)

        df = df[mask].copy()

        if len(df) == 0:
            print("Error: No data in specified time range", file=sys.stderr)
            raise SystemExit(2)

        print(f"Filtered to {len(df)} rows (time range: {df['time'].min():.1f}s - {df['time'].max():.1f}s)")
    else:
        print(f"Loaded {len(df)} rows (time range: {df['time'].min():.1f}s - {df['time'].max():.1f}s)")

    # Prefer plotting against the original timestamp column (readable time-of-day x-axis).
    x = pd.to_datetime(df["timestamp"])
    tz = getattr(x.dt, "tz", None)
    tz_label = f" ({tz})" if tz is not None else ""

    plot_time_series(
        df,
        x=x,
        x_label=f"time of day{tz_label}",
        autosize=True,
    )


def analyze_sampling_cli():
    """CLI entry point for analyzing sampling characteristics."""
    parser = argparse.ArgumentParser(
        description="Analyze sampling characteristics of time series data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze with auto-detected schema
  analyze-sampling data.csv

  # Specify schema explicitly
  analyze-sampling data.csv --schema vakaros_atlas_telemetry

  # Custom gap threshold (in seconds)
  analyze-sampling data.csv --gap-threshold 2.0
        """
    )
    parser.add_argument(
        "filename",
        help="Input CSV filename"
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="Schema name (default: auto-detect)"
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=None,
        help="Gap threshold in seconds (default: 3x median interval)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output (useful for scripting)"
    )

    args = parser.parse_args()

    from pathlib import Path

    # Validate file exists
    if not Path(args.filename).exists():
        print(f"Error: File '{args.filename}' not found", file=sys.stderr)
        sys.exit(1)

    try:
        # Load data
        result = load_file(args.filename, schema=args.schema)
        df = result.df

        if "time" not in df.columns:
            print(f"Error: File '{args.filename}' does not contain 'time' column", file=sys.stderr)
            sys.exit(1)

        times = df["time"].values

        # Analyze sampling
        results = _analyze_sampling(times, gap_threshold=args.gap_threshold)

        if not args.quiet:
            _print_sampling_results(results, args.filename)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _print_sampling_results(results: dict, filename: str) -> None:
    """Print formatted sampling analysis results."""
    from pathlib import Path

    print(f"\nSampling Analysis: {Path(filename).name}")
    print("=" * 60)

    # Basic info
    print(f"\nBasic Statistics:")
    print(f"  Total samples: {results['total_samples']:,}")
    print(f"  Time range: {results['time_range'][0]:.2f}s to {results['time_range'][1]:.2f}s")
    print(f"  Duration: {results['duration']:.2f}s")

    # Sample rate
    print(f"\nUpdate Frequency:")
    if results['sample_rate']:
        print(f"  Detected sample rate: {results['sample_rate']:.2f} Hz")
    else:
        print(f"  Detected sample rate: irregular")

    # Interval statistics
    stats = results['interval_stats']
    print(f"\nSample Interval Statistics:")
    mean_hz = 1/stats['mean'] if stats['mean'] > 0 else float('inf')
    median_hz = 1/stats['median'] if stats['median'] > 0 else float('inf')
    print(f"  Mean:   {stats['mean']:.4f}s ({mean_hz:.2f} Hz)")
    print(f"  Median: {stats['median']:.4f}s ({median_hz:.2f} Hz)")
    print(f"  Std dev: {stats['std']:.4f}s")
    print(f"  Min:    {stats['min']:.4f}s")
    print(f"  Max:    {stats['max']:.4f}s")

    # Regularity assessment
    cv = results['regularity_cv']
    print(f"\nSampling Regularity:")
    print(f"  Coefficient of variation (CV = std/mean): {cv:.4f}")
    if cv < 0.01:
        regularity = "Highly regular"
    elif cv < 0.05:
        regularity = "Mostly regular"
    elif cv < 0.1:
        regularity = "Somewhat irregular"
    else:
        regularity = "Irregular"
    print(f"  Assessment: {regularity}")

    # Gaps
    gaps = results['gaps']
    print(f"\nGaps:")
    if gaps:
        print(f"  Detected: {len(gaps)} significant gap(s)")
        print(f"  First 5 gaps:")
        for i, (start, end, size) in enumerate(gaps[:5]):
            print(f"    Gap {i+1}: idx {start}->{end}, size {size:.2f}s")
        if len(gaps) > 5:
            print(f"    ... and {len(gaps) - 5} more gap(s)")
    else:
        print(f"  No significant gaps detected")

    print("=" * 60)
    print()
