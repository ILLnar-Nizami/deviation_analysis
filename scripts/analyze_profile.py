import pstats
import sys


def analyze_profile(stats_file='profile.stats', limit=20):
    """Analyze and print profile statistics."""
    p = pstats.Stats(stats_file)

    print("\nCumulative time:")
    p.sort_stats('cumulative').print_stats(limit)

    print("\nTime per call:")
    p.sort_stats('time').print_stats(limit)

    print("\nCall count:")
    p.sort_stats('calls').print_stats(limit)


if __name__ == '__main__':
    stats_file = sys.argv[1] if len(sys.argv) > 1 else 'profile.stats'
    analyze_profile(stats_file)
