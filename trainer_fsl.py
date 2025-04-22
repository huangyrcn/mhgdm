import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_utils import Dataset
from utils.metric import EuclideanDistances


class Trainer:
    def __init__(self, args):
        self.args = args
        self.epoch_num = args.epoch_num
        self.P_num = args.P_num
        # self.N_way = args.N_way
        self.K_shot = args.K_shot
        self.query_size = args.query_size
        self.eval_interval = args.eval_interval
        self.test_task_num = args.test_task_num

        self.dataset = Dataset(args.dataset_name, args)
        args.train_classes_num = self.dataset.train_classes_num
        args.node_fea_size = self.dataset.train_graphs[0].node_features.shape[1]
        args.sample_input_size = (args.gin_layer - 1) * args.gin_hid

        args.N_way = self.dataset.test_classes_num
        self.N_way = self.dataset.test_classes_num

        self.baseline_mode = args.baseline_mode

        self.model = Model(args).cuda()

        self.N_sample_prob = (
            np.ones([self.dataset.train_classes_num]) / self.dataset.train_classes_num
        )

        self.use_loss_based_prob = args.use_loss_based_prob
        self.loss_based_prob = torch.ones([100, self.dataset.train_classes_num]).cuda() * 10

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=args.weight_decay
        )

        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        best_test_acc = 0
        best_valid_acc = 0

        train_accs = []
        for i in range(self.epoch_num):
            loss, acc, class_loss = self.train_one_step(mode="train", epoch=i)

            train_accs.append(np.mean(acc))

            if loss == None:
                continue

            if i % 50 == 0:
                self.scheduler.step()
                print(
                    "Epoch {} Train Acc {:.4f} Loss {:.4f} Class Loss {:.4f}".format(
                        i, np.mean(train_accs), loss, class_loss
                    )
                )
                f.write(
                    "Epoch {} Train Acc {:.4f} Loss {:.4f} Class Loss {:.4f}".format(
                        i, np.mean(train_accs), loss, class_loss
                    )
                    + "\n"
                )

            if i % self.eval_interval == 0:
                with torch.no_grad():
                    test_accs = []
                    start_test_idx = 0
                    while (
                        start_test_idx
                        < len(self.dataset.test_graphs)
                        - self.K_shot * self.dataset.test_classes_num
                    ):
                        loss, test_acc, class_loss = self.train_one_step(
                            mode="test", epoch=i, test_idx=start_test_idx
                        )
                        if loss == None:
                            continue
                        test_accs.extend(test_acc.tolist())
                        start_test_idx += self.N_way * self.query_size

                    print("test task num", len(test_accs))
                    mean_acc = sum(test_accs) / len(test_accs)
                    if mean_acc > best_test_acc:
                        best_test_acc = mean_acc

                    print(
                        "Mean Test Acc {:.4f}  Best Test Acc {:.4f}".format(mean_acc, best_test_acc)
                    )
                    f.write(
                        "Mean Test Acc {:.4f}  Best Test Acc {:.4f}".format(mean_acc, best_test_acc)
                        + "\n"
                    )

                    test_accs = []
                    start_test_idx = 0
                    while (
                        start_test_idx
                        < len(self.dataset.validation_graphs)
                        - self.K_shot * self.dataset.train_classes_num
                    ):
                        loss, test_acc, class_loss = self.train_one_step(
                            mode="valid", epoch=i, test_idx=start_test_idx
                        )
                        if loss == None:
                            continue
                        test_accs.extend(test_acc.tolist())
                        start_test_idx += self.N_way * self.query_size

                    print("test task num", len(test_accs))
                    mean_acc = sum(test_accs) / len(test_accs)
                    if mean_acc > best_valid_acc:
                        best_valid_acc = mean_acc

                    print(
                        "Mean Valid Acc {:.4f}  Best Valid Acc {:.4f}".format(
                            mean_acc, best_valid_acc
                        )
                    )
                    f.write(
                        "Mean Valid Acc {:.4f}  Best Valid Acc {:.4f}".format(
                            mean_acc, best_valid_acc
                        )
                        + "\n"
                    )

        return best_test_acc

    def train_one_step(self, mode, epoch, test_idx=None, baseline_mode=None):
        if mode == "train":
            self.model.train()
            if self.use_loss_based_prob:
                p = (
                    (self.loss_based_prob - (self.loss_based_prob - 20).relu())
                    .mean(0)
                    .softmax(-1)
                    .cpu()
                    .detach()
                    .numpy()
                )
                if epoch % 50 == 0:
                    print(self.loss_based_prob.mean(0))
                if np.isnan(p).sum() > 0:
                    print(self.loss_based_prob)
                    return None, None, None
            else:
                p = self.N_sample_prob

            first_N_class_sample = np.random.choice(
                list(range(self.dataset.train_classes_num)), self.N_way, p=p, replace=False
            )
            current_task = self.dataset.sample_one_task(
                self.dataset.train_tasks,
                first_N_class_sample,
                K_shot=self.K_shot,
                query_size=self.query_size,
            )
        elif mode == "test":
            self.model.eval()
            first_N_class_sample = np.array(list(range(self.dataset.test_classes_num)))
            current_task = self.dataset.sample_one_task(
                self.dataset.test_tasks,
                first_N_class_sample,
                K_shot=self.K_shot,
                query_size=self.query_size,
                test_start_idx=test_idx,
            )
        elif mode == "valid":
            self.model.eval()
            first_N_class_sample = np.array(list(range(self.dataset.test_classes_num)))
            current_task = self.dataset.sample_one_task(
                self.dataset.test_tasks,
                first_N_class_sample,
                K_shot=self.K_shot,
                query_size=self.query_size,
                test_start_idx=test_idx,
            )

        if self.baseline_mode == "proto" or self.baseline_mode == "relation":

            current_sample_input_embs, current_sample_input_embs_selected = (
                self.model.sample_input_GNN([current_task])
            )  # [N(K+Q), emb_size]

            input_embs = current_sample_input_embs.reshape(
                [self.N_way, self.K_shot + self.query_size, -1]
            )
            support_embs = input_embs[:, : self.K_shot, :]
            query_embs = input_embs[:, self.K_shot :, :]  # [N, q, emb_size]

            support_protos = support_embs.mean(1)  # [N, emb_size]

            if self.baseline_mode == "proto":
                scores = -EuclideanDistances(
                    query_embs.reshape([self.N_way * self.query_size, -1]), support_protos
                )
            elif self.baseline_mode == "relation":
                scores = self.model.rel_classifier(
                    torch.cat(
                        [
                            support_protos.unsqueeze(1)
                            .repeat([1, self.query_size, 1])
                            .reshape(self.N_way * self.query_size, -1),
                            query_embs.reshape([self.N_way * self.query_size, -1]),
                        ],
                        dim=-1,
                    )
                )

            if mode == "train":
                label = torch.tensor(np.array(list(range(self.N_way)))).cuda()
                label = label.unsqueeze(0).repeat([self.query_size, 1]).t()
                label = label.reshape([self.N_way * self.query_size])

            else:
                labels = []
                for graphs in current_task["query_set"]:
                    labels.append(torch.tensor(np.array([graph.label for graph in graphs])))
                label = torch.cat(labels, -1).cuda()

            y_preds = torch.argmax(scores, dim=1)

            if current_task["append_count"] != 0:
                scores = scores[: label.shape[0] - current_task["append_count"], :]
                y_preds = y_preds[: label.shape[0] - current_task["append_count"]]
                label = label[: label.shape[0] - current_task["append_count"]]

            acc = (y_preds == label).float().cpu().numpy()
            loss = self.criterion(scores, label)

            if mode == "train":

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            return loss, acc, 0

        # --calculte similarities (conduct base classification)
        current_sample_input_embs, current_sample_input_embs_selected = self.model.sample_input_GNN(
            [current_task]
        )  # [N(K+Q), emb_size]

        classifiy_result = self.model.base_classifier(
            current_sample_input_embs.reshape(
                [self.N_way, self.K_shot + self.query_size, self.model.sample_input_emb_size]
            ).mean(1)
        )  # [N, N]

        loss_type = nn.CrossEntropyLoss(reduction="none")

        class_loss = loss_type(classifiy_result, torch.tensor(first_N_class_sample).cuda())

        if torch.isnan(class_loss).sum() > 0:
            print(current_sample_input_embs)
            print(class_loss)
            print(classifiy_result)
            print(first_N_class_sample)
            print(1 / 0)
            return None, None, None

        if self.use_loss_based_prob and mode == "train":
            self.loss_based_prob[epoch % 100, first_N_class_sample] = class_loss
            if torch.isnan(self.loss_based_prob).sum() > 0:
                return None, None, None

        sim_matrix = classifiy_result.softmax(-1)

        sample_rate = sim_matrix.sum(0).softmax(-1).cpu().detach().numpy()

        exclude_self = False
        if exclude_self and mode == "train":
            sample_rate[first_N_class_sample] = 0

        P_tasks, support_classes = self.dataset.sample_P_tasks(
            self.dataset.train_tasks,
            self.P_num,
            (sample_rate / sample_rate.sum()),
            N_way=self.N_way,
            K_shot=self.K_shot,
            query_size=self.query_size,
        )

        test_sample_test = False
        if test_sample_test and mode == "test":

            P_tasks, support_classes = self.dataset.sample_P_tasks(
                self.dataset.test_tasks,
                self.P_num,
                np.ones([self.dataset.test_classes_num]) / self.dataset.test_classes_num,
                N_way=self.N_way,
                K_shot=self.K_shot,
                query_size=self.query_size,
            )

        support_classes = [first_N_class_sample] + support_classes

        total_tasks = [current_task] + P_tasks
        sample_input_embs, sample_input_embs_selected = self.model.sample_input_GNN(
            total_tasks
        )  # [(P+1)NK, emb_size]

        sample_output_embs_query, proto_input_embs = self.model.construct_sample_graph(
            sample_input_embs, support_classes, sample_input_embs_selected
        )
        # split the sample_output as support and query

        proto_output_embs, task_input_embs = self.model.construct_proto_graph(
            proto_input_embs, support_classes
        )
        task_output_embs = self.model.construct_task_graph(task_input_embs, support_classes)

        # --final classification

        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        scores = self.model.classify_tasks(
            sample_output_embs_query, proto_input_embs, task_output_embs
        )  # [(P+1)NQ, N]

        if mode == "train":
            label = torch.tensor(np.array(list(range(self.N_way)))).cuda()
            label = label.unsqueeze(0).repeat([self.query_size, 1]).t()
            label = (
                label.unsqueeze(0)
                .repeat([(self.P_num + 1), 1, 1])
                .reshape([(self.P_num + 1) * self.N_way * self.query_size])
            )

            current_only = False
            if current_only:
                label = label.reshape([self.P_num + 1, self.N_way, self.query_size])[
                    0, :, :
                ].reshape(self.N_way * self.query_size)
                scores = scores.reshape([self.P_num + 1, self.N_way, self.query_size, self.N_way])[
                    0, :, :, :
                ].reshape(self.N_way * self.query_size, self.N_way)

        else:
            labels = []
            for graphs in current_task["query_set"]:
                labels.append(torch.tensor(np.array([graph.label for graph in graphs])))
            label = torch.cat(labels, -1).cuda()
            scores = scores.reshape([self.P_num + 1, self.N_way, self.query_size, self.N_way])[
                0, :, :, :
            ].reshape(self.N_way * self.query_size, self.N_way)

        y_preds = torch.argmax(scores, dim=1)

        if current_task["append_count"] != 0:
            scores = scores[: label.shape[0] - current_task["append_count"], :]
            y_preds = y_preds[: label.shape[0] - current_task["append_count"]]
            label = label[: label.shape[0] - current_task["append_count"]]

        acc = (y_preds == label).float().cpu().numpy()
        loss = self.criterion(scores, label) + class_loss.mean() * 0.001

        if mode == "train":

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, acc, class_loss.mean()
