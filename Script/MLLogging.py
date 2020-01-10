import logging

class DispatchingFormatter:
    def __init__(self, formatters, default_formatter):
        self._formatters = formatters
        self._default_formatter = default_formatter

    def format(self, record):
        formatter = self._formatters.get(record.name, self._default_formatter)
        return formatter.format(record)

class Logger:
    level_dict = {
        'NOTSET'  : logging.NOTSET,
        'DEBUG'   : logging.DEBUG,
        'INFO'    : logging.INFO,
        'WARNING' : logging.WARNING,
        'ERROR'   : logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }

    ChangeFrom = DispatchingFormatter(
            { 'c' : logging.Formatter( '[%(asctime)s] [%(levelname)-4s]: %(message)s', '%Y-%m-%d %H:%M:%S'),
              'p' : logging.Formatter( '[%(levelname)-4s]: %(message)s'),
              'n' : logging.Formatter( '%(message)s' ),
            }, 
            logging.Formatter('%(message)s')
     )

    def __init__(self, outpath, filemode='w',  clevel = 'INFO', Flevel = 'INFO'):

        logging.basicConfig(
            level    = Logger.level_dict[clevel] ,
            format   = '[%(asctime)s] [%(levelname)-4s]: %(message)s',
            datefmt  = '%Y-%m-%d %H:%M:%S',
            filename = None,
        )

        File = logging.FileHandler(outpath,  mode= filemode)
        File.setLevel(Logger.level_dict[Flevel])
        File.setFormatter(Logger.ChangeFrom)
        logging.getLogger().addHandler(File)
        #Hand = logging.StreamHandler()
        #Hand.setFormatter(Logger.ChangeFrom)
        #logging.getLogger().addHandler(Hand)

        self.R = logging
        self.C = logging.getLogger('c')
        self.P = logging.getLogger('p')
        self.N = logging.getLogger('n')
        self.CIF = logging.getLogger('c').info
        self.NIF = logging.getLogger('n').info
        self.CWA = logging.getLogger('c').warning
        self.NWA = logging.getLogger('n').warning