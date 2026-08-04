from .cli.main import main
import sys
sys.set_int_max_str_digits(0)

if __name__ == '__main__':
    raise SystemExit(main())
