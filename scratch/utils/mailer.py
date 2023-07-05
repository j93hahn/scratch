import os
import logging
from contextlib import contextmanager
import traceback
import smtplib
from email.message import EmailMessage

logger = logging.getLogger(__name__)


def send_email(
    subject, body, to,
    sender="jjahn@exp.ttic.edu",
):
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = to

    # Send the message via our own SMTP server.
    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()


"""
CONDA_DEFAULT_ENV                   # multinerf
SLURMD_NODENAME                     # gpu32
os.getcwd().split('/')[-1]          # 23_0703_1812_09 - experiment folder name
"""

@contextmanager
def exp_fail(to="jjahn@uchicago.edu", subject=None, do_send=True):
    """note that contextmanager can be used as a decorator as well"""
    if subject is None:
        subject = f"Experiment {os.getcwd().split('/')[-1]} failed"
    try:
        yield
    except Exception:
        logger.info(f"Exception triggered. Emailing {to}")
        if do_send:
            send_email(subject=subject, body=traceback.format_exc(), to=to)
    finally:
        pass


if __name__ == "__main__":
    # send_email("test", "test", "jjahn@uchicago.edu")
    with exp_fail():
        error_msg = f"experiment {os.getcwd().split('/')[-1]} failed on {os.environ.get('SLURMD_NODENAME')} with conda env {os.environ.get('CONDA_DEFAULT_ENV')}"
        raise Exception(error_msg)
