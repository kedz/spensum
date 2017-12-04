import torch



class CentroidMixIn(object):
    def zero_identity_mask(self, batch_size, input_size, b1, b2, fill_value=1.0):
        ''' 
        Compute a mask doc_size x doc_size mask where all values are 1 
        except along the diagonal which is 0. 
        This mask is repeated batch_size times to create a 
        batch_size x doc_size x doc_size mask tensor.
        '''
        mask = b1.resize_(input_size, input_size)
        mask = mask.fill_(fill_value) - \
            torch.diag(b2.resize_(input_size).fill_(fill_value))
        mask_tensor = mask.view(1, input_size, input_size).repeat(
            batch_size, 1, 1)
        return torch.autograd.Variable(mask_tensor)

    #TODO Test the shit out of this!!!
    def centroid(self, inputs, mask, remove_identity=False):
        batch_size = inputs.embedding.size(0)
        input_size = inputs.embedding.size(1)

        b1 = inputs.embedding.data.new()
        b2 = inputs.embedding.data.new()

        if remove_identity:
            zi_mask = self.zero_identity_mask(batch_size, input_size, b1, b2)

            batch_mask1 = mask.view(
                batch_size, 1, input_size).repeat(1, input_size, 1)
            zi_mask = zi_mask.masked_fill(batch_mask1, 0)
            #batch_mask2 = mask.view(
            #    batch_size, input_size, 1).repeat(1, 1, input_size)
            #zi_mask = zi_mask.masked_fill(batch_mask2, 0)
            weights = (1 / zi_mask.sum(2))
            weights = weights.masked_fill(mask, 0)
            weights = weights.view(batch_size, 1, input_size)
            weights = weights.repeat(1, input_size, 1)
            weights.data.mul_(zi_mask.data)

            batch_mask2 = mask.view(
                batch_size, input_size, 1).repeat(1, 1, input_size)
            weights = weights.masked_fill(batch_mask2, 0)
            centroid = weights.bmm(inputs.embedding).squeeze(1)

        else:
            raise NotImplementedError("Need to implement masked mean.")
#            weights = (1 / (1 - mask.float()).sum(1)) 
#            weights = weights.view(batch_size, 1)
#            weights = weights.repeat(1, input_size)
#            weights = weights.masked_fill(mask, 0)
#            weights = weights.view(batch_size, 1, input_size)  
#            print(weights)
#            exit()
#            #weights = torch.autograd.Variable(weights.data)

        return centroid
