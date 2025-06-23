import pandas as pd

import numpy as np
import pandas as pd
import galois


class qrcode:
    def __init__(self, version=1, mode='byte', level="l", mask=0, rs_encoder=None) -> None:
        self.version = version
        self.mode = mode
        self.level = level
        self.mask = mask
        self.field = galois.GF(2**8, repr='power')
        self.ec_codeword = lambda i, g: (i % g) if rs_encoder is None else rs_encoder

    def __str__(self) -> str:
        return f"QR(version={self.version}, mode={self.mode}, level={self.level}, size={self.shape})"
    
    # general properties
    @property
    def n_modules(self):
        return 4 * self.version + 17
    
    @property
    def shape(self):
        return (self.n_modules, self.n_modules)
    
    # version
    @property
    def version(self):
        return self._version
    
    @version.setter
    def version(self, version):
        if int(version) not in versions:
            raise ValueError(f'version ({version}) must be an integer between {versions[0]} and {versions[-1]}.')
        self._version = int(version)
    
    # encoding mode
    @property
    def mode(self):
        return self._mode
    
    @mode.setter
    def mode(self, mode):
        if mode.lower() not in modes:
            raise ValueError(f'mode ({mode}) must be one of {list(modes)}.')
        self._mode = mode.lower()

    # correction level
    @property
    def level(self):
        return self._level
    
    @level.setter
    def level(self, level):
        if level.lower() not in levels:
            raise ValueError(f'level ({level.lower()}) must be in {levels}')
        self._level = level.lower()

    # number of error codewords per block
    @property
    def n_err(self):
        pass

    # blocks
    @property
    def group_sizes(self):
        return groups[["group_size_0", "group_size_1"]].loc[self.version, self.level].to_numpy()
    
    @property
    def block_sizes(self):
        return groups[["block_data_size_0", "block_data_size_1"]].loc[self.version, self.level].to_numpy()

    # mask pattern
    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        if mask not in masks:
            raise ValueError(f'mask ({mask}) must be an integer in {masks}.')
        self._mask = mask
    
    @property
    def mask_pattern(self):
        if self.mask == 0:  # 000
            return lambda i, j: (i + j) % 2 == 0
        if self.mask == 1:  # 001
            return lambda i, j: i % 2 == 0
        if self.mask == 2:  # 010
            return lambda i, j: j % 3 == 0
        if self.mask == 3:  # 011
            return lambda i, j: (i + j) % 3 == 0
        if self.mask == 4:  # 100
            return lambda i, j: ((i // 2) + (j // 3)) % 2 == 0
        if self.mask == 5:  # 101
            return lambda i, j: (i * j) % 2 + (i * j) % 3 == 0
        if self.mask == 6:  # 110
            return lambda i, j: ((i * j) % 2 + (i * j) % 3) % 2 == 0
        if self.mask == 7:  # 111
            return lambda i, j: ((i * j) % 3 + (i + j) % 2) % 2 == 0
        
    # finders
    @property
    def finder_positions(self):
        return [(0, 0), (0, self.n_modules-7), ((self.n_modules-7, 0))]
    
    @property
    def finder_shape(self):
        return 7 * np.ones(2).astype(int)
    
    @property
    def finder_area(self):
        modules = np.zeros(self.shape)
        # mask finders
        for pos in self.finder_positions:
            modules[pos[0]:pos[0] + self.finder_shape[0], pos[1]:pos[1] + self.finder_shape[1]] = 1
        return modules

    @property
    def finder_pattern(self):
        modules = np.zeros(self.shape)
        
        # build 1 finder pattern
        finder = np.ones(self.finder_shape)
        finder[1:6, 1:6] = np.zeros(self.finder_shape - 2)
        finder[2:5, 2:5] = np.ones(self.finder_shape - 4)

        # fill the corners
        for (x, y) in self.finder_positions:
            x_start, x_stop = x, x + self.finder_shape[0]
            y_start, y_stop = y, y + self.finder_shape[1]
            modules[x_start:x_stop, y_start:y_stop] = finder
        return modules
    
    # separators
    @property
    def separator_positions(self):
        pos = [(7, i) for i in range(self.n_modules) if i < 8 or i > self.n_modules - 9]
        pos += [(i, 7) for i in range(self.n_modules) if i < 8 or i > self.n_modules - 9]
        pos += [(self.n_modules - 8, i) for i in range(self.n_modules) if i < 8]
        pos += [(i, self.n_modules - 8) for i in range(self.n_modules) if i < 8]
        return pos
    
    @property
    def separator_area(self):
        modules = np.zeros(self.shape)
        for pos in self.separator_positions:
            modules[pos] = 1
        return modules
    
    @property
    def separator_pattern(self):
        modules = np.zeros(self.shape)
        for pos in self.separator_positions:
            modules[pos] = 0
        return modules
    
    # alignments
    @property
    def alignment_positions(self):
        if self.version == 1:
            return []
        else:
            row = np.array(alignments["alignment_positions"][self.version]) - 2
            pos = [
                    (x, y)
                    for i, x in enumerate(row)
                        for j, y in enumerate(row)
                            if (i, j) not in [(0, 0), (0, row.size - 1), (row.size - 1, 0)]
                ]
        return pos
    
    @property
    def alignment_shape(self):
        return 5 * np.ones(2).astype(int)
    
    @property
    def alignment_area(self):
        area = np.zeros(self.shape)
        for pos in self.alignment_positions:
            area[pos[0]:pos[0] + self.alignment_shape[0], pos[1]:pos[1] + self.alignment_shape[1]] = 1
        return area
    
    @property
    def alignment_pattern(self):
        modules = np.zeros(self.shape)
        
        # build 1 alignment pattern 
        alignment = np.ones(self.alignment_shape)
        alignment[1:4, 1:4] = np.zeros((3, 3))
        alignment[2, 2] = 1

        # position the alignment patterns
        for pos in self.alignment_positions:
            modules[pos[0]:pos[0] + self.alignment_shape[0], pos[1]:pos[1] + self.alignment_shape[1]] = alignment
        return modules
    
    # timings
    @property
    def timing_position(self):
        pos = [(i, 6) for i in range(8, self.n_modules-8)]
        # pos += [(6, i) for i in range(8, self.n_modules-8)]
        return pos
    
    @property
    def timing_length(self):
        return self.n_modules - 2 * 8
    
    @property
    def timing_area(self):
        modules = np.zeros(self.shape)
        for pos in self.timing_position:
            modules[pos] = modules[pos[::-1]] = 1
        return modules

    @property
    def timing_pattern(self):
        # initialize pattern
        modules = np.zeros(self.shape)

        for i, pos in enumerate(self.timing_position):
            modules[pos] = modules[pos[::-1]] = (i + 1) % 2
        return modules
    
    # dark pattern
    @property
    def dark_position(self):
        return (4 * self.version + 9, 8)
    
    @property
    def dark_area(self):
        modules = np.zeros(self.shape)
        modules[self.dark_position] = 1
        return modules
    
    @property
    def dark_pattern(self):
        pattern = np.zeros(self.shape)
        pattern[self.dark_position] = 1
        return pattern
        
    # format
    @property
    def format_positions(self):
        h_pos = [
            (8, j) 
            for j in range(self.n_modules) 
            if (0 <= j <= 5) or (7 <= j <= 7) or (self.n_modules - 8 <= j)
        ]
        v_pos = [
            (i, 8) 
            for i in range(self.n_modules) 
            if (0 <= i <= 5) or (7 <= i <= 8) or (self.n_modules - 7 <= i)
        ]
        return *h_pos, *v_pos[::-1]
    
    @property
    def format_string(self):
        return format_strings[self.mask][self.level]
    
    @property
    def format_area(self):
        modules = np.zeros(self.shape)
        pos = self.format_positions
        for i in range(len(self.format_string)):
            modules[pos[i]] = 1
            modules[pos[i + 15]] = 1
        return modules

    @property
    def format_pattern(self):
        pattern = np.zeros(self.shape)
        pos = self.format_positions
        for (i, b) in enumerate(self.format_string):
            pattern[pos[i]] = int(b)
            pattern[pos[i+15]] = int(b)        
        return pattern
    
    # version information
    @property
    def versinfo_string(self):
        if self.version >= 7:
            return versinfos["versinfo"][self.version]

    @property
    def versinfo_position(self):
        return (0, self.n_modules - 11)
    
    @property
    def versinfo_shape(self):
        return (6, 3)
    
    @property
    def versinfo_area(self):
        modules = np.zeros(self.shape)
        if self.versinfo_string is not None:
            modules[self.versinfo_position[0]:self.versinfo_position[0]+self.versinfo_shape[0], self.versinfo_position[1]:self.versinfo_position[1]+self.versinfo_shape[1]] = 1
            modules[self.versinfo_position[1]:self.versinfo_position[1]+self.versinfo_shape[1], self.versinfo_position[0]:self.versinfo_position[0]+self.versinfo_shape[0]] = 1
        return modules

    @property
    def versinfo_pattern(self):
        modules = np.zeros(self.shape)

        if self.versinfo_string is not None:
            versinfo = np.array(list(map(int, self.versinfo_string[::-1]))).reshape(self.versinfo_shape)
            # print(versinfo)
            
            modules[self.versinfo_position[1]:self.versinfo_position[1]+self.versinfo_shape[1], self.versinfo_position[0]:self.versinfo_position[0]+self.versinfo_shape[0]] = versinfo.T

            modules[self.versinfo_position[0]:self.versinfo_position[0]+self.versinfo_shape[0], self.versinfo_position[1]:self.versinfo_position[1]+self.versinfo_shape[1]] = versinfo
        return modules

    # data
    @property
    def data_area(self):
        modules = np.ones(self.shape)
        modules -= self.finder_area
        modules -= self.separator_area 
        modules -= np.minimum(1, (self.timing_area + self.alignment_area)) # timing and alignment areas overlap
        modules -= self.format_area
        modules -= self.dark_area
        modules -= self.versinfo_area
        return modules

    @property
    def data_length(self):
        return int(self.data_area.sum())
    
    def data_pattern(self, data):
        if len(data) != self.data_length:
            raise ValueError(f'size of data ({np.size(data)}) does not fit the data space ({self.data_length}).')
        
        modules = np.zeros(self.shape)
        data_area = self.data_area
        module = dict(x=self.n_modules - 1, y=self.n_modules - 1)
        i, step_x = 0, -1
        while i < len(data):
            # print(module)
            if data_area[module['x'], module['y']] > 0:
                # fill free module with a data bit
                modules[module['x'], module['y']] = int(data[i])
                i += 1
            
            # update module and direction
            # if module['y'] == 0:
            #     # we reached the first column and so simply fill it up or down.
            #     module['x'] += step_x
            if module['y'] == 6:
                # we reached the vertical timing
                module['y'] -= 1
            elif (module['y'] % 2) == int(module['y'] < 6):
                # the current column is even so we move to the column on the left.
                module['y'] -= 1
            else: # module['y'] % 2 == 1:
                # the current column is odd.
                if (module['x'] == 0 and step_x < 0) or (module['x'] == (self.n_modules - 1) and step_x > 0):
                    # We are either at the top or the bottom of the current column (which is not the first one).
                    # So we move by one column on the left and change the filling direction.
                    module['y'] -= 1
                    step_x *= -1
                else:
                    # we are between the top and the bottom of the current column (which is not the first one).
                    # we move to the next module diagonally, either upward or downward depending on the filling direction.
                    module['y'] += 1 
                    module['x'] += step_x
        return modules

    def data_cci(self, data):
        cci = len(data) // cci_lengths.loc[self.version, self.mode]
        # print(cci)
        return f'{cci:08b}'
    
    def prepare_data(self, information):
        data = mode_codes['code'][self.mode]
        data += self.data_cci(information)
        data += information

        # number of data bits that must be filled
        n_data_bits = groups["n_data_codewords"][self.version, self.level] * 8

        # add terminator sequence
        if len(data) < n_data_bits:
            data += '0' * min(4, n_data_bits - len(data))
        
        # pad with zeros to fill the last incomplete byte (if any)
        data += '0' * (len(data) % 8)

        # add padding bytes if necessary
        data += "1110110000010001" * ((n_data_bits - len(data)) // 16) + "11101100" *(((n_data_bits - len(data)) % 16) // 8)
        return data
    
    def prepare_blocks(self, data):
        # number of codewords per blocks
        n_ec_codewords_per_blocks = groups.loc[self.version, self.level]["n_ec_codewords_per_block"]


        # block sizes
        # print(self.block_sizes, self.group_sizes)
        data_block_sizes = [0] + [self.block_sizes[0]] * self.group_sizes[0] + [self.block_sizes[1]] * self.group_sizes[1]
        # print(data_block_sizes)

        # compute the limits of the data block
        data_blocks_limits = np.cumsum(data_block_sizes) * 8
        # print(data_blocks_limits)
        
        # split data in bytes
        data_blocks_byte = [
            [data[i:i + 8] for i in range(start, stop, 8)]
                for start, stop in zip(data_blocks_limits[:-1], data_blocks_limits[1:])
        ]
        # print(list(map(len, data_blocks_byte)))
        # print(np.array(data_blocks_byte))

        # convert data bytes to ints
        data_blocks_int = [
            [int(byte, 2) for byte in block]
                for block in data_blocks_byte
        ]
        # print(data_blocks_int)

        # data galois polynomials
        data_blocks_poly = [
            galois.Poly(block, field=self.field)
            for block in data_blocks_int
        ]
        # print(data_blocks_poly)

        # generator galois polynomial
        g = self.data_generator_polynomial(n_ec_codewords_per_blocks)
        # print(f'g(X) = {g}')

        # shift galois polynomial
        shift = galois.Poly([1] + [0] * (g.degree), field=self.field)
        # print(f'shift(X) = {shift}')

        # ec galois polynomials
        ec_blocks_poly = [
            self.ec_codeword(data_block_poly * shift, g)
            for data_block_poly in data_blocks_poly
        ]
        # print(ec_blocks_poly)

        # convert from ec ints to bytes
        ec_blocks_byte = [
            [f'{codeword:08b}' for codeword in ec_block_poly.coeffs]
                for ec_block_poly in ec_blocks_poly
        ]
        # print(list(map(len, ec_blocks_byte)))
        # print(np.array(ec_blocks_byte))

        
        # interleave data codewords
        encoded_data = ""
        for codeword in range(self.block_sizes.max()):
            for block in range(self.group_sizes.sum()):
                    if codeword < data_block_sizes[block + 1]:
                        # # print(block, codeword)
                        encoded_data += data_blocks_byte[block][codeword]
        
        # interleave ec codewords
        for codeword in range(n_ec_codewords_per_blocks):
            for block in range(self.group_sizes.sum()):
                    encoded_data += ec_blocks_byte[block][codeword]
        
        # pad with 0s to fill the data length
        pad = self.data_length - len(encoded_data)
        # print(pad)
        encoded_data += '0' * pad
        # print(encoded_data)

        return encoded_data

    def data_generator_polynomial(self, degree):
        g = galois.Poly([1, self.field.primitive_element ** 0], field=self.field)
        for i in range(1, degree):
            g *= galois.Poly([1, self.field.primitive_element ** i], field=self.field)
            # print(f'g(X) = {g}')
        return g
    
    # qr code builder
    def __call__(self, information):
        data = self.prepare_data(information)
        encoded_data = self.prepare_blocks(data)

        modules = np.zeros(self.shape)
        modules += self.data_pattern(encoded_data)

        data_area = self.data_area
        for i in range(modules.shape[0]):
            for j in range(modules.shape[1]):
                if data_area[i, j] > 0:
                    modules[i, j] += self.mask_pattern(i, j)
                    modules[i, j] %= 2

        modules += self.finder_pattern
        modules += self.separator_pattern
        modules += self.alignment_pattern
        modules += self.timing_pattern
        modules += self.dark_pattern
        modules += self.format_pattern
        modules += self.versinfo_pattern
        
        modules = (modules > 0).astype(int)
        return modules

