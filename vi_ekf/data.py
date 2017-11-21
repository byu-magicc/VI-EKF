import cPickle
import numpy as np
import data_loader
import cv2
import klt_tracker
import cPickle

class Data(object):
    def __init__(self):
        self.time = np.linspace(0, 1, 100)
        self.R = {'alt': 0.01,
                  'acc': np.diag([0.01, 0.01]),
                  'att': np.diag([0.0001, 0.0001, 0.0001]),
                  'pos': np.diag([0.0001,0.0001, 0.0001]),
                  'zeta': np.diag([0.0001, 0.0001]),
                  'depth': 0.01}

    def indexer(self, target_time, source_time):
        index_for_target = []
        current_index = 0
        for t in target_time:
            while current_index < len(source_time) and source_time[current_index] <= t:
                current_index += 1
            index_for_target.append(current_index - 1)

        assert len(index_for_target) == len(target_time)

        return index_for_target

    def __getitem__(self, item):
        if item >= len(self):
            raise IndexError

        t = 0.
        dt = 0.
        pos = np.zeros((3, 1))
        vel = np.zeros((3, 1))
        att = np.zeros((4, 1))
        gyro = np.zeros((3, 1))
        acc = np.zeros((3, 1))
        zetas = []
        depths = []
        ids = []

        return t, dt, pos, vel, att, gyro, acc, zetas, depths, ids

    def __len__(self):
        return 0

    @property
    def x0(self):
        return np.zeros((17, 1))

    def __test__(self):
        assert self.x0.shape == (17,1), self.x0.shape
        time = self.start
        for x in self:
            assert len(x) == 12
            t, dt, pos, vel, att, gyro, acc, b_w, b_a, zetas, depths, ids = x
            time += dt
            assert time == t, (time, t, dt)
            assert all([gyro is None, acc is None]) or not all([gyro is None, acc is None])
            assert type(zetas) == type(depths) == type(ids) == list
            assert len(zetas) == len(depths) == len(ids)
            assert type(t) == float or type(t) == np.float64, type(t)
            assert type(dt) == float or type(t) == np.float64, type(t)
            assert ((pos.shape == (3, 1)) if pos is not None else True), pos.shape
            assert (vel.shape == (3, 1)) if vel is not None else True
            assert (att.shape == (4, 1)) if att is not None else True
            assert (gyro.shape == (3, 1)) if gyro is not None else True
            assert (acc.shape == (3, 1)) if acc is not None else True
            assert (b_w.shape == (3, 1)) if acc is not None else True
            assert (b_a.shape == (3, 1)) if acc is not None else True
            assert zetas[0].shape == (3, 1) if len(zetas) > 0 else True, zetas[0].shape
            assert depths[0].shape == (1, 1) if len(depths) > 0 else True, depths[0].shape
            assert all([type(id) == int or type(id) == np.int64 for id in ids]), type(ids[0])


class FakeData(Data):
    def __init__(self, start=-1, end=np.inf):
        super(FakeData, self).__init__()
        self.data = cPickle.load(open('generated_data.pkl', 'rb'))
        self.s = np.argmax(np.array(self.data['truth_NED']['t']) > start)
        self.time = self.data['imu_data']['t'][self.s:]

        self.truth_indexer = self.indexer(self.time, self.data['truth_NED']['t'])
        self.imu_indexer = self.indexer(self.time, self.data['imu_data']['t'])
        self.feature_indexer = self.indexer(self.time, self.data['features']['t'])

    @property
    def x0(self):
        return np.concatenate(list(self[0][2:5]) +
                              [np.zeros((3, 1)),
                               np.zeros((3, 1)),
                               0.2*np.ones((1, 1))], axis=0)

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError

        t = self.time[i]
        dt = self.time[0] if i == 0 else (self.time[i] - self.time[i - 1])
        pos = self.data['truth_NED']['pos'][self.truth_indexer[i], None].T
        vel = self.data['truth_NED']['vel'][self.truth_indexer[i], None].T
        att = self.data['truth_NED']['att'][self.truth_indexer[i], None].T
        gyro = self.data['imu_data']['gyro'][self.imu_indexer[i], None].T
        acc = self.data['imu_data']['acc'][self.imu_indexer[i], None].T
        zetas = list(np.swapaxes(self.data['features']['zeta'][self.feature_indexer[i], :, None], 1, 2))
        depths = list(self.data['features']['depth'][self.feature_indexer[i], :, None, None])
        ids = range(len(zetas))

        return t, dt, pos, vel, att, gyro, acc, zetas, depths, ids

    def __len__(self):
        return len(self.time)


class ETHData(Data):
    def __init__(self, filename='/mnt/pccfs/not_backed_up/eurocmav/V1_01_easy/mav0', start=-1, end=np.inf, sim_features=False, load_new=False):
        super(ETHData, self).__init__()
        if load_new:
            self.data = data_loader.load_data(filename, start, end, sim_features)
            cPickle.dump(self.data, open('data/generated_data.pkl', 'wb'))
        else:
            self.data = cPickle.load(open('data/generated_data.pkl', 'rb'))

        self.time = np.unique(np.concatenate([self.data['imu'][:, 0]]))
                                              # self.data['truth'][:, 0],
                                              # self.data['feat_time']]))
        self.time = self.time[(self.time > start) & (self.time < end)]

        self.truth_indexer = self.indexer(self.time, self.data['truth'][:, 0])
        self.imu_indexer = self.indexer(self.time, self.data['imu'][:, 0])
        self.feature_indexer = self.indexer(self.time, self.data['feat_time'])
        self.start = start

        # self.tracker = klt_tracker.KLT_tracker(25)
        # self.undistort, P = data_loader.make_undistort_funtion(intrinsics=self.data['cam0_sensor']['intrinsics'],
        #                                                     resolution=self.data['cam0_sensor']['resolution'],
        #                                                     distortion_coefficients=self.data['cam0_sensor']['distortion_coefficients'])

        # self.inverse_projection = np.linalg.inv(P)

    def compute_features(self, image):
        image = self.undistort(image)[..., None]

        zetas, ids = [], []
        lambdas, ids = self.tracker.load_image(image)

        if lambdas is not None and len(lambdas) > 0:
            lambdas = np.pad(lambdas[:, 0], [(0, 0), (0, 1)], 'constant', constant_values=0)

            zetas = self.inverse_projection.dot(lambdas.T).T[..., None]
            zetas /= np.sqrt((zetas * zetas).sum(axis=1, keepdims=True))

        return list(zetas), list(ids)

    @property
    def x0(self):
        return np.concatenate([self.data['truth'][self.truth_indexer[0], 1:4, None],
                               self.data['truth'][self.truth_indexer[0], 8:11, None],
                               self.data['truth'][self.truth_indexer[0], 4:8, None],
                               self.data['truth'][self.truth_indexer[0], 14:17, None],
                               self.data['truth'][self.truth_indexer[0], 11:14, None],
                               2.5*0.2*np.ones((1, 1))], axis=0)

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError

        t = self.time[i]
        dt = self.time[0] - self.start if i == 0 else (self.time[i] - self.time[i - 1])
        pos, vel, att, gyro, acc = None, None, None, None, None
        zetas, ids, depths = [], [], []

        # if self.truth_indexer[i] - self.truth_indexer[i - 1] != 0:
        pos = self.data['truth'][self.truth_indexer[i], 1:4, None]
        vel = self.data['truth'][self.truth_indexer[i], 8:11, None]
        att = self.data['truth'][self.truth_indexer[i], 4:8, None]
        b_w = self.data['truth'][self.truth_indexer[i], 11:14, None]
        b_a = self.data['truth'][self.truth_indexer[i], 14:17, None]


        if self.imu_indexer[i] - self.imu_indexer[i - 1] != 0:
            gyro = self.data['imu'][self.imu_indexer[i], 1:4, None]
            acc = self.data['imu'][self.imu_indexer[i], 4:7, None]

        if self.feature_indexer[i] - self.feature_indexer[i - 1] != 0:
            ids = list(self.data['ids'][self.feature_indexer[i], :])
            zetas = []
            depths = []
            for feat, l in enumerate(ids):
                zetas.append(self.data['zetas'][feat][self.feature_indexer[i], :, None])
                depths.append(self.data['depths'][feat][self.feature_indexer[i], :, None])

            # image = cv2.imread(self.data['cam0_frame_filenames'][self.feature_indexer[i]], cv2.IMREAD_GRAYSCALE)
            # zetas, ids = self.compute_features(image)

        return t, dt, pos, vel, att, gyro, acc, b_w, b_a, zetas, depths, ids

    def __len__(self):
        return len(self.time)

    def __test__(self):
        # image = np.zeros([480, 752, 1]).astype(np.uint8)
        # size = 100
        # image[480//2 - size:480//2 + size, 752//2 - size:752//2 + size] = 255
        # zeta, id = self.compute_features(image)
        #
        # assert len(zeta) == 4
        # assert len(zeta) == len(id), (len(zeta), len(id))
        # assert False, 'we should manually calculate and check the correct zetas'

        super(ETHData, self).__test__()


if __name__ == '__main__':
    d = FakeData()
    d.__test__()
    print 'Passed FakeData tests.'

    d = ETHData()
    d.__test__()
    print 'Passed ETHData tests.'
