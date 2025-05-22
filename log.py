import logging
import sys


from args import args

args.log_file = f"{args.checkpoint}/{args.log_file}"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        logging.FileHandler(args.log_file),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger()

if args.resume:
    logger.info(f"Resume training... from {args.resume}")

